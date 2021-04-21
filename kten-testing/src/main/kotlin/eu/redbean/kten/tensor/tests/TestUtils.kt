package eu.redbean.kten.tensor.tests

import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.serialization.serializers.tensorFromJson
import org.opentest4j.AssertionFailedError
import kotlin.math.abs


fun assertTensorEquals(expected: Tensor, actual: Tensor?, tolerance: Float = 1e-6f) {
    fun equalsFail() {
        throw AssertionFailedError("expected: <${expected}> but was: <${actual}>", expected, actual)
    }

    if (actual == null) {
        equalsFail()
    } else {
        if (expected.shape != actual.shape)
            equalsFail()

        val volume = expected.shape.fold(1, Int::times)
        val flatExpected = expected.reshape(volume)
        val flatActual = actual.reshape(volume)

        for (i in 0 until volume) {
            if (abs(flatExpected.getValue(listOf(i)) - flatActual.getValue(listOf(i))) > tolerance)
                equalsFail()
        }
    }
}

fun loadTensorFromJson(resourcePath: String) = Tensor.tensorFromJson(object {}.javaClass.getResource(resourcePath).readText())
