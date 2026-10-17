"""Rol funcional de cada peso de un SavedModel P3 2D-US, derivado de su grafo.

Por qué hace falta
------------------
`--init` copiaba los pesos "por forma en orden" (`pool.setdefault(shape)`). Eso
funciona solo si el orden en que el SavedModel lista sus variables coincide con
el de `build_model()`. No coincide: el SavedModel base de Keras 2 lista las dos
ramas INTERCALADAS por nivel y con la rama F2 primero,

    (2,1,10) -> [F2 L1 Tanh, F2 L1 Sigmoid, F1 L1 Tanh, F1 L1 Sigmoid]

mientras `build_model()` crea la rama A = F1 y luego la B = F2. Como todas las
formas coinciden entre ramas y niveles, el emparejado por forma asignaba a cada
brazo los pesos del OTRO (48 de 62 pesos), y el fine-tune no era un fine-tune.
Medido: BCE 1,0433 frente a 0,3524 del modelo base (peor que aleatorio).

Cómo se deduce el rol
---------------------
Del grafo, no de los nombres de las variables:

  * rama: a qué mitad del cross llega cada conv. Se lee de la operación que parte
    la entrada: `Split`/`SplitV` (puerto 0 = 1ª mitad = F1, 1 = F2, es el caso
    del SavedModel de Keras 2) o `StridedSlice`/`Slice` (por el offset del
    `begin`, es el caso de un re-export de Keras 3 con `t[:, :64]` / `t[:, 64:]`).
  * nivel: número de convs de rama aguas arriba (1 en el primer bloque gated).
  * activación: nodo `Tanh`/`Sigmoid` que consume la salida de la conv.
  * bloques Dense: cadena `MatMul -> activación -> Mul` desde el concat.

Así funciona igual con el SavedModel original de Keras 2 (`conv1d_52..77`, orden
intercalado) y con un re-export de Keras 3 (`conv1d`, `conv1d_1`, ...), lo que
permite encadenar fine-tunes sin tocar nada.

Claves de rol ('|' para no colisionar con nombres de capa):

    conv|<arm>|<level>|<act>|<k|b>      conv|F1|1|t|k
    fus|<arm>|<k|b>                     fus|F2|b
    dense|<block>|<act>|<k|b>           dense|0|t|k
    out|<k|b>                           out|b

Autochequeo: si alguna variable del base queda sin clasificar o se clasifica dos
veces, se lanza ValueError en vez de copiar mal en silencio.
"""
import numpy as np

SPLIT_OPS = ("Split", "SplitV")
SLICE_OPS = ("StridedSlice", "Slice")
CONV_OP = "Conv2D"
DENSE_OP = "MatMul"
ACT_OPS = ("Tanh", "Sigmoid")
PASS_OPS = ("Squeeze", "Reshape", "ExpandDims", "Identity", "Cast", "BiasAdd")


def _node_name(tensor):
    return tensor.split(":")[0]


def _port(tensor):
    last = tensor.split(":")[-1]
    return int(last) if last.isdigit() else 0


def _const_ints(node):
    """Valores enteros de un nodo Const (int_val/int64_val/tensor_content)."""
    t = node.attr["value"].tensor
    if t.int_val:
        return np.array(t.int_val, dtype=np.int64)
    if t.int64_val:
        return np.array(t.int64_val, dtype=np.int64)
    if t.tensor_content:
        return np.frombuffer(t.tensor_content, dtype=np.int64) if t.dtype == 9 \
            else np.frombuffer(t.tensor_content, dtype=np.int32).astype(np.int64)
    return np.array([], dtype=np.int64)


class _Graph:
    """Vistas mínimas sobre el FunctionDef del signature."""

    def __init__(self, fn):
        self.fn = fn
        self.nodes = {n.name: n for n in fn.node_def}
        self.consumers = {}
        for n in fn.node_def:
            for t in n.input:
                self.consumers.setdefault((_node_name(t), _port(t)), []).append(n)

    def src(self, tensor):
        return self.nodes.get(_node_name(tensor))

    def first_input(self, node):
        return node.input[0] if node.input else None

    def backward(self, tensor, stops, limit=64):
        """Sube aguas arriba hasta un nodo de `stops`.

        Devuelve (cadena, nodo_parada, tensor_de_salida_de_ese_nodo).
        """
        chain, cur = [], tensor
        for _ in range(limit):
            n = self.src(cur)
            if n is None:
                return chain, None, None
            if n.op in stops:
                return chain, n, cur
            chain.append(n)
            nxt = self.first_input(n)
            if nxt is None:
                return chain, None, None
            cur = nxt
        raise ValueError("grafo: cadena aguas arriba demasiado larga")

    def forward_first(self, tensor, wanted, limit=6):
        """Primer nodo de op en `wanted` alcanzado aguas abajo vía PASS_OPS."""
        seen, frontier = set(), [(_node_name(tensor), _port(tensor))]
        for _ in range(limit):
            nxt = []
            for key in frontier:
                for c in self.consumers.get(key, []):
                    if c.name in seen:
                        continue
                    seen.add(c.name)
                    if c.op in wanted:
                        return c
                    if c.op in PASS_OPS:
                        nxt.append((c.name, 0))
            frontier = nxt
            if not frontier:
                return None
        return None

    def activation_of(self, node):
        act = self.forward_first(node.name, ACT_OPS)
        return act.op if act is not None else None

    def bias_adder(self, node):
        """Nodo que suma el sesgo: BiasAdd, o Add/AddV2 con una variable.

        Keras 2 emite BiasAdd para todas las capas; Keras 3 emite AddV2 en la
        capa de salida (ops.add) y puede emitir un Add residual en otras
        arquitecturas, por eso solo se acepta el Add cuyo segundo operando sale
        de un ReadVariableOp.
        """
        seen, frontier = set(), [(_node_name(node.name), 0)]
        for _ in range(6):
            nxt = []
            for key in frontier:
                for c in self.consumers.get(key, []):
                    if c.name in seen:
                        continue
                    seen.add(c.name)
                    if c.op in ("BiasAdd", "Add", "AddV2") and len(c.input) > 1 \
                            and self.read_variable(c.input[1]) is not None:
                        return c
                    if c.op in PASS_OPS:
                        nxt.append((c.name, 0))
            frontier = nxt
            if not frontier:
                return None
        return None

    def read_variable(self, tensor, limit=8):
        """Sube aguas arriba hasta el ReadVariableOp y devuelve su recurso."""
        cur = tensor
        for _ in range(limit):
            n = self.src(cur)
            if n is None:
                return None
            if n.op == "ReadVariableOp":
                return n.input[0]
            nxt = self.first_input(n)
            if nxt is None:
                return None
            cur = nxt
        return None

    def arm_of(self, tensor, half_hint=64):
        """'F1' o 'F2' según la mitad del cross de la que sale `tensor`."""
        _, node, out = self.backward(tensor, SPLIT_OPS + SLICE_OPS)
        if node is None:
            raise ValueError(
                f"{_node_name(tensor)}: no encuentro el corte de la entrada "
                f"({SPLIT_OPS + SLICE_OPS}) aguas arriba")
        if node.op in SPLIT_OPS:
            n_out = int(node.attr["num_split"].i) if "num_split" in node.attr else 2
            if n_out != 2:
                raise ValueError(f"{node.name}: split de {n_out} salidas, se esperan 2")
            port = _port(out)
            if port not in (0, 1):
                raise ValueError(f"{node.name}: puerto inesperado {port}")
            return "F1" if port == 0 else "F2"
        begin_node = self.src(node.input[1])
        if begin_node is None or begin_node.op != "Const":
            raise ValueError(f"{node.name}: no puedo leer el begin del slice")
        begin = _const_ints(begin_node)
        # el eje que se parte es el del cross (no el batch): busco el primer
        # offset distinto de 0 en la posición 1 (forma (batch, 128, canales))
        offset = int(begin[1]) if len(begin) > 1 else int(begin[0])
        if offset not in (0, half_hint):
            raise ValueError(f"{node.name}: offset {offset} inesperado (se espera 0 o {half_hint})")
        return "F1" if offset == 0 else "F2"


def _symbolic_name(tensor):
    """Nombre de un tensor simbólico (None si es eager)."""
    try:
        return tensor.name
    except AttributeError:
        return getattr(tensor, "_name", None)


def _resource_binding(sm, sig, fn):
    """{nombre_de_recurso_del_grafo: tf.Variable}, sin mirar nombres de capa.

    Los SavedModels capturan sus variables como recursos. La cadena exacta es:

        captures[i]                        -> (tensor simbólico 'unknown_i', handle)
        call.input[i]                      -> ese mismo tensor simbólico
        fn.signature.input_arg[i].name     -> nombre del recurso que usan los
                                              ReadVariableOp dentro de la función

    El emparejado es por identidad del handle y orden posicional, no por parecido
    de nombres. Por eso sirve tanto para el SavedModel de Keras 2 (variables
    intercaladas por nivel, nombres conv1d_52..77) como para un re-export de
    Keras 3 (nombres de nodo con sufijos de desambiguación), donde un emparejado
    por nombre es ambiguo: 'conv1d_1' y 'conv1d_1_2' son capas distintas.
    """
    handles = {id(v.handle): v for v in sm.variables}
    by_symbolic = {}
    for pair in sig.graph.captures:
        handle, symbolic = None, None
        for x in pair:
            if id(x) in handles:
                handle = handles[id(x)]
            name = _symbolic_name(x)
            if name is not None and name.startswith("unknown"):
                symbolic = _node_name(name)
        if symbolic is not None and handle is not None:
            by_symbolic[symbolic] = handle.name.split(":")[0]

    calls = [n for n in sig.graph.as_graph_def().node
             if n.op in ("StatefulPartitionedCall", "PartitionedCall")]
    if len(calls) != 1:
        raise ValueError(f"el signature tiene {len(calls)} llamadas, se espera 1")
    args = [a.name for a in fn.signature.input_arg]
    if len(args) != len(calls[0].input):
        raise ValueError(
            f"la función tiene {len(args)} entradas y la llamada {len(calls[0].input)}")
    out = {}
    for i, tensor in enumerate(calls[0].input):
        name = by_symbolic.get(_node_name(tensor))
        if name is not None:
            out[args[i]] = name
    if not out:
        raise ValueError("no pude emparejar los recursos del grafo con las variables")
    return out


def _bind_variable(graph, node, resources, part, trace):
    """Nombre de la variable que lee `node` como filtro (`kernel`) o sesgo (`bias`)."""
    owner = node
    if part == "bias":
        badd = graph.bias_adder(node)
        if badd is None:
            raise ValueError(f"{node.name}: no encuentro el nodo que suma su sesgo")
        resource = graph.read_variable(badd.input[1])
        owner = badd
    else:
        resource = graph.read_variable(node.input[1])
    if resource is None:
        raise ValueError(f"{owner.name}: no encuentro su ReadVariableOp")
    name = resources.get(_node_name(resource))
    if name is None:
        raise ValueError(
            f"{owner.name}: el recurso {_node_name(resource)} no es una variable "
            "conocida del SavedModel")
    trace[name] = trace.get(name, 0) + 1
    return name


def _analyse(saved_model_path):
    """(roles, variables) del SavedModel. Ver `derive_roles`."""
    import tensorflow as tf  # import perezoso: los tests puros no necesitan TF

    sm = tf.saved_model.load(saved_model_path)
    if not sm.signatures:
        raise ValueError(f"{saved_model_path}: el SavedModel no expone signatures")
    sig = sm.signatures.get("serving_default") or next(iter(sm.signatures.values()))
    funcs = list(sig.graph.as_graph_def().library.function)
    if not funcs:
        raise ValueError(f"{saved_model_path}: el signature no expone FunctionDef")
    graph = _Graph(max(funcs, key=lambda f: len(f.node_def)))

    variables = {v.name.split(":")[0]: v for v in sm.variables}
    if not variables:
        raise ValueError(f"{saved_model_path}: el SavedModel no tiene variables")
    resources = _resource_binding(sm, sig, graph.fn)

    roles, trace = {}, {}
    convs = [n for n in graph.fn.node_def if n.op == CONV_OP]
    if not convs:
        raise ValueError(f"{saved_model_path}: no hay Conv2D en el grafo")

    for conv in convs:
        kern = _bind_variable(graph, conv, resources, "kernel", trace)
        width = int(variables[kern].shape[0])
        arm = graph.arm_of(conv.input[0])
        if width == 1:                                    # convolución 1x1 de fusión
            roles[f"fus|{arm}|k"] = kern
            roles[f"fus|{arm}|b"] = _bind_variable(graph, conv, resources, "bias", trace)
            continue
        if width != 2:
            raise ValueError(f"{conv.name}: kernel de ancho {width}, se esperaba 1 o 2")
        chain, _, _ = graph.backward(conv.input[0], SPLIT_OPS + SLICE_OPS)
        # cuento convs de rama aguas arriba: la primera no tiene ninguna
        level = 1 + sum(1 for c in chain if c.op == CONV_OP)
        act = graph.activation_of(conv)
        if act is None:
            raise ValueError(f"{conv.name}: sin Tanh/Sigmoid aguas abajo")
        key = "t" if act == "Tanh" else "s"
        roles[f"conv|{arm}|{level}|{key}|k"] = kern
        roles[f"conv|{arm}|{level}|{key}|b"] = _bind_variable(
            graph, conv, resources, "bias", trace)

    concats = [n for n in graph.fn.node_def if n.op in ("Concat", "ConcatV2")]
    if len(concats) != 1:
        raise ValueError(f"{saved_model_path}: se esperaba 1 Concat, hay {len(concats)}")
    concat = concats[0]
    order = [graph.arm_of(t) for t in concat.input[:2]]
    if order != ["F1", "F2"]:
        raise ValueError(
            f"{saved_model_path}: el concat va en orden {order}, no [F1, F2]; "
            "el modelo nuevo espera la rama F1 primero")

    def denses_after(tensor):
        found, frontier, seen = [], [(_node_name(tensor), _port(tensor))], set()
        for _ in range(4):
            nxt = []
            for key in frontier:
                for c in graph.consumers.get(key, []):
                    if c.name in seen:
                        continue
                    seen.add(c.name)
                    if c.op == DENSE_OP:
                        found.append(c)
                    elif c.op in PASS_OPS:
                        nxt.append((c.name, 0))
            frontier = nxt
        return found

    def act_key(node):
        act = graph.activation_of(node)
        if act is None:
            raise ValueError(f"{node.name}: sin activación")
        return "t" if act == "Tanh" else "s"

    def product_after(nodes):
        """El Mul que multiplica las dos ramas gated: Tanh * Sigmoid."""
        acts = set()
        for n in nodes:
            act = graph.forward_first(n.name, ACT_OPS)
            if act is None:
                raise ValueError(f"{n.name}: sin activación")
            acts.add(act.name)
        for node in graph.fn.node_def:
            if node.op == "Mul" and {_node_name(i) for i in node.input} == acts:
                return node
        raise ValueError(f"no encuentro el Mul que combina {sorted(acts)}")

    def block(matmul_nodes, block_index):
        for mm in matmul_nodes:
            key = act_key(mm)
            roles[f"dense|{block_index}|{key}|k"] = _bind_variable(
                graph, mm, resources, "kernel", trace)
            roles[f"dense|{block_index}|{key}|b"] = _bind_variable(
                graph, mm, resources, "bias", trace)

    first = denses_after(concat.name)
    if len(first) != 2:
        raise ValueError(f"{saved_model_path}: bloque Dense 0 con {len(first)} MatMul")
    block(first, 0)
    second = denses_after(product_after(first).name)
    if len(second) != 2:
        raise ValueError(f"{saved_model_path}: bloque Dense 1 con {len(second)} MatMul")
    block(second, 1)
    tail = denses_after(product_after(second).name)
    if len(tail) != 1:
        raise ValueError(f"{saved_model_path}: se esperaba 1 MatMul de salida, hay {len(tail)}")
    roles["out|k"] = _bind_variable(graph, tail[0], resources, "kernel", trace)
    roles["out|b"] = _bind_variable(graph, tail[0], resources, "bias", trace)

    repeated = {k: v for k, v in trace.items() if v > 1}
    if repeated:
        raise ValueError(f"{saved_model_path}: variables asignadas más de una vez: {repeated}")
    unused = sorted(set(variables) - set(trace))
    if unused:
        raise ValueError(f"{saved_model_path}: variables sin rol: {unused}")
    return roles, variables


def derive_roles(saved_model_path):
    """{clave_de_rol: nombre_de_variable} para un SavedModel P2-US 2D.

    Claves descritas en el docstring del módulo. Lanza ValueError si el grafo no
    tiene la forma esperada o si alguna variable queda sin clasificar.
    """
    return _analyse(saved_model_path)[0]


def assign_weights(model, saved_model_path):
    """Copia los pesos del SavedModel base al modelo nuevo POR ROL.

    `model` debe traer `model.weight_roles` (lo pone `build_model` en
    scripts/train_model.py). Devuelve (copiados, aleatorios). Lanza ValueError si
    algún peso del modelo nuevo no tiene equivalente: preferimos fallar a copiar
    mal en silencio.
    """
    roles = getattr(model, "weight_roles", None)
    if roles is None:
        raise ValueError(
            "el modelo no trae weight_roles: constrúyelo con build_model() de "
            "scripts/train_model.py")
    if len(model.weights) != len(roles):
        raise ValueError(
            f"el modelo tiene {len(model.weights)} pesos y {len(roles)} roles")
    base, variables = _analyse(saved_model_path)
    copied, missing = 0, []
    for w, role in zip(model.weights, roles):
        name = base.get(role)
        if name is None:
            missing.append(role)
            continue
        w.assign(variables[name].numpy())
        copied += 1
    if missing:
        raise ValueError(
            f"sin peso base equivalente para: {sorted(set(missing))} "
            f"(¿otra arquitectura en {saved_model_path}?)")
    return copied, len(model.weights) - copied
