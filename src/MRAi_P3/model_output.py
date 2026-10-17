"""Salida unica de un modelo, sin depender del nombre de la capa.

Los SavedModels originales (Keras 2) exponen la ultima Dense con su nombre
(`dense_14` en 2D, `dense_9` en 3D). Los re-exportados con Keras 3
(`model.export`, como hace `scripts/train_model.py`) la exponen como
`output_0`. Ese nombre es un detalle de serializacion, no parte del contrato
del modelo: los modelos P3 tienen una unica salida y los consumidores deben
tomarla sea cual sea su clave.

Este modulo no importa TensorFlow a proposito: es logica pura y se testea sin
el entorno de entrenamiento.
"""


def single_output(pred):
    """Devuelve la unica salida de `model.predict`.

    Parameters
    ----------
    pred : dict | array
        Lo que devuelve `model.predict`: un dict `{nombre_capa: tensor}` para
        los modelos servidos con `TFSMLayer`, o directamente el array si el
        modelo no expone salidas estructuradas.

    Returns
    -------
    array
        La unica salida del modelo, sin mirar su clave.

    Raises
    ------
    ValueError
        Si el modelo expone mas de una salida: el nombre es ambiguo y elegir
        una arbitrariamente daria predicciones silenciosamente incorrectas.
    """
    if not isinstance(pred, dict):
        return pred
    if len(pred) != 1:
        raise ValueError(
            "se esperaba una unica salida del modelo, hay "
            f"{len(pred)}: {sorted(pred)}")
    return next(iter(pred.values()))
