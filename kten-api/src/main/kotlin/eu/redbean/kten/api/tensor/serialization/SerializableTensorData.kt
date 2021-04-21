package eu.redbean.kten.api.tensor.serialization

import eu.redbean.kten.api.autograd.utils.toStoreSize
import java.io.Serializable

data class SerializableTensorData(
    val shape: List<Int>,
    val data: FloatArray
) : Serializable {
    init {
        if (shape.isEmpty())
            throw IllegalStateException("Shape of tensor data cannot be empty")

        if (shape.contains(0))
            throw IllegalStateException("Shape cannot have dimension with 0 size, but got tensor data with shape: ${shape}")

        if (shape.toStoreSize() != data.size)
            throw IllegalStateException("Invalid serializable tensor data, data size and size implied by shape does not match, " +
                    "data size: ${data.size}, shape: ${shape}, shape implied size: ${shape.toStoreSize()}")
    }
}