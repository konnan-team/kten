package eu.redbean.kten.api.tensor.serialization

interface TensorSerializer<T> {

    fun serialize(commonTensorDescriptor: CommonSerializableTensorDescriptor): T

    fun deserialize(serializedValue: T): CommonSerializableTensorDescriptor

}