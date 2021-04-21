package eu.redbean.kten.api.tensor.serialization.serializers

import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.serialization.TensorSerializer
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream

object BinaryTensorSerializer: TensorSerializer<ByteArray> {

    override fun serialize(commonTensorDescriptor: CommonSerializableTensorDescriptor): ByteArray {
        val baos = ByteArrayOutputStream()
        ObjectOutputStream(baos).use { it.writeObject(commonTensorDescriptor) }
        return baos.toByteArray()
    }

    override fun deserialize(serializedValue: ByteArray): CommonSerializableTensorDescriptor {
        val bais = ByteArrayInputStream(serializedValue)
        ObjectInputStream(bais).use {
            return it.readObject() as CommonSerializableTensorDescriptor
        }
    }
}

fun Tensor.Companion.tensorFromBinary(bytes: ByteArray): Tensor {
    return deserializeWith(BinaryTensorSerializer, bytes)
}

fun Tensor.toBinary(): ByteArray {
    return serializeWith(BinaryTensorSerializer)
}
