package eu.redbean.kten.api.tensor.serialization

import java.io.Serializable

data class CommonSerializableTensorDescriptor(
    val data: SerializableTensorData,
    val gradientData: SerializableTensorData?
): Serializable {
    init {
        if (gradientData != null && data.shape != gradientData.shape)
            throw IllegalStateException("Invalid tensor descriptor, data shape and gradient shape must match, " +
                    "but got data with shape: ${data.shape} and gradient with shape: ${gradientData.shape}")
    }
}