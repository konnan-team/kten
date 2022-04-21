package eu.redbean.kten.tensor.tests.nn

import eu.redbean.kten.api.autograd.functions.nn.maxPooling1d
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.tensor.tests.assertTensorEquals
import eu.redbean.kten.tensor.tests.testcases.TestCase
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertAll
import java.time.Duration

abstract class PoolingTestsBase {

    @Test
    fun should_calculate_max_pooling_for_1d_cases() {
        poolingTestCase("/maxpool1d_k3_s2_p1_d2_on_2x3x10_ceil.xml") {
            it.maxPooling1d(kernel = 3, stride = 2, padding = 1, dilation = 2, useCeil = true)
        }
    }

    fun poolingTestCase(testCasePath: String, op: (Tensor) -> Tensor) {
        val testCase = TestCase.loadTestCase(testCasePath)

        val t1 = testCase["t1"]!!.asVariable(requiresGrad = true)
        val expectedRes = testCase["res"]!!
        val expectedResSum = testCase["sumRes"]!!
        val expectedT1Grad = testCase["t1Grad"]!!

        val time = System.nanoTime()

        val res = op(t1)
        val resSum = Tensor.sum(res)
        resSum.backward()

        val duration = Duration.ofNanos(System.nanoTime() - time)
        println("exec time: ${duration.toMillis()}ms")

        assertAll(
            { assertTensorEquals(expectedRes, res, 1e-4f) },
            { assertTensorEquals(expectedResSum, resSum, 1e-4f) },
            { assertTensorEquals(expectedT1Grad, t1.grad(), 1e-4f) },
        )
    }

}