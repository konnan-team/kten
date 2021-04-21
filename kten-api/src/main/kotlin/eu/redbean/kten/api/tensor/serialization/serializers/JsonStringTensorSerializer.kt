package eu.redbean.kten.api.tensor.serialization.serializers

import eu.redbean.kten.api.autograd.utils.tensorIndexing
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.serialization.SerializableTensorData
import eu.redbean.kten.api.tensor.serialization.TensorSerializer

object JsonStringTensorSerializer: TensorSerializer<String> {

    override fun serialize(commonTensorDescriptor: CommonSerializableTensorDescriptor): String {
        var res = """{"data":${tensorDataToJsonArray(commonTensorDescriptor.data)}"""
        if (commonTensorDescriptor.gradientData != null) {
            res += ""","gradient":${tensorDataToJsonArray(commonTensorDescriptor.gradientData)}"""
        }
        return "$res}"
    }

    private fun tensorDataToJsonArray(tensorData: SerializableTensorData): String {
        val strBuilder = StringBuilder()

        val shape = tensorData.shape
        val size = shape.size
        val data = tensorData.data

        for (i in 0 until size)
            strBuilder.append("[")

        val reversedShape = shape.reversed()
        val indices = IntArray(size)
        for (it in 0 until shape.fold(1, Int::times)) {
            var i = it
            var dimensionIndex = size - 1
            for (dimension in reversedShape) {
                indices[dimensionIndex] = i % dimension
                i /= dimension
                dimensionIndex--
            }
            if (it > 0 && indices.last() == 0) {
                for (j in indices.size - 1 downTo 0) {
                    if (indices[j] == 0)
                        strBuilder.append("]")
                    else {
                        strBuilder.append(",")
                        for (k in 0 until indices.size - j - 1)
                            strBuilder.append("[")
                        break
                    }
                }
            }
            if (indices.last() != 0)
                strBuilder.append(",")

            strBuilder.append(data[shape.tensorIndexing(indices).storeIndex])
        }
        for (i in 0 until size)
            strBuilder.append("]")

        return strBuilder.toString()
    }

    override fun deserialize(serializedValue: String): CommonSerializableTensorDescriptor {
        val strValue = serializedValue.replace("\\s".toRegex(), "")
        val dataIndex = strValue.indexOf("\"data\":", ignoreCase = true)

        if (dataIndex == -1)
            throw IllegalArgumentException("Invalid json - no data property found")

        val gradIndex = strValue.indexOf("\"gradient\":", ignoreCase = true)

        val objectEndIndex = strValue.indexOf("}")
        val dataEndIndex = if (gradIndex == -1)
            objectEndIndex
        else if (gradIndex > dataIndex)
            gradIndex - 1
        else
            objectEndIndex

        val dataString = strValue.substring(dataIndex + 7 until dataEndIndex)

        val data = tensorDataFromJsonArray(dataString)
        var grad: SerializableTensorData? = null

        if (gradIndex != -1) {
            val gradEndIndex = if (gradIndex > dataIndex) objectEndIndex else dataIndex - 1
            val gradString = strValue.substring(gradIndex + 11 until gradEndIndex)
            grad = tensorDataFromJsonArray(gradString)
        }

        return CommonSerializableTensorDescriptor(data, grad)
    }

    fun tensorDataFromJsonArray(jsonArray: String): SerializableTensorData {
        var depth = 0
        val shape = mutableListOf<Int>()
        val strValues = mutableListOf<String>()
        var valueString = ""
        var valuesDimDone = false
        for (i in jsonArray.indices) {
            val current = jsonArray[i]
            when {
                current == '[' -> {
                    depth++
                    if (shape.size < depth)
                        shape.add(0)
                }
                current in '0'..'9' || current in listOf('.', '-', 'e', 'E') -> valueString += current
                current == ']' -> {
                    if (valueString.isNotEmpty()) {
                        strValues += valueString
                        valueString = ""
                        if (!valuesDimDone)
                            shape[depth - 1] += 1
                        valuesDimDone = true
                    }
                    if (depth != shape.size && (depth == 1 || shape[depth-2] == 0))
                        shape[depth - 1] += 1
                    depth--
                }
                current == ',' -> {
                    if (valueString.isNotEmpty()) {
                        strValues += valueString
                        valueString = ""
                        if (!valuesDimDone)
                            shape[depth - 1] += 1
                    } else {
                        if (depth != shape.size && (depth == 1 || shape[depth-2] == 0))
                            shape[depth - 1] += 1
                    }
                }
                else -> throw IllegalArgumentException("Invalid json - invalid character: ${current} in array")
            }
        }

        val data = strValues.map {
            val strValue = if (it.endsWith(".")) it + "0" else it
            strValue.toFloat()
        }.toFloatArray()

        return SerializableTensorData(shape, data)
    }

}

fun Tensor.Companion.tensorFromJson(jsonString: String): Tensor {
    return deserializeWith(JsonStringTensorSerializer, jsonString)
}

fun Tensor.toJson(): String {
    return serializeWith(JsonStringTensorSerializer)
}
