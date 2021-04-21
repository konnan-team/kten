package eu.redbean.kten.tensor.tests.nn

import eu.redbean.kten.api.autograd.functions.nn.conv1d
import eu.redbean.kten.api.autograd.functions.nn.conv2d
import eu.redbean.kten.api.autograd.functions.nn.conv2dTranspose
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.Tensor.Companion.tensorOf
import eu.redbean.kten.tensor.tests.assertTensorEquals
import eu.redbean.kten.tensor.tests.loadTensorFromJson
import eu.redbean.kten.tensor.tests.testcases.TestCase
import org.junit.jupiter.api.RepeatedTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertAll
import java.time.Duration

abstract class ConvolutionTestsBase {

    @RepeatedTest(20)
    fun should_calculate_conv1d_pad0_dilation1_stride1_with_grads() {
        val t1 = tensorOf(
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            11, 12, 13, 14, 15, 16, 17, 18, 19,

            21, 22, 23, 24, 25, 26, 27, 28, 29,
            31, 32, 33, 34, 35, 36, 37, 38, 39
        ).reshape(2, 2, 9).asVariable(requiresGrad = true)

        val weight = tensorOf(
            1, 2, 1,
            2, 1, 2,

            1, 2, 1,
            3, 4, 3,

            2, 3, 2,
            4, 3, 4
        ).reshape(3, 2, 3).asVariable(requiresGrad = true)

        val bias = tensorOf(3, 2, 1, requiresGrad = true)

        val res = t1.conv1d(weight, bias)

        assertTensorEquals(
            tensorOf(
                71,  80,  89,  98, 107, 116, 125,
                130, 144, 158, 172, 186, 200, 214,
                147, 165, 183, 201, 219, 237, 255,

                251, 260, 269, 278, 287, 296, 305,
                410, 424, 438, 452, 466, 480, 494,
                507, 525, 543, 561, 579, 597, 615
            ).reshape(2, 3, 7),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                4, 11, 15, 15, 15, 15, 15, 11,  4,
                9, 17, 26, 26, 26, 26, 26, 17,  9,

                4, 11, 15, 15, 15, 15, 15, 11,  4,
                9, 17, 26, 26, 26, 26, 26, 17,  9
            ).reshape(2, 2, 9),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(
                196, 210, 224,
                336, 350, 364,

                196, 210, 224,
                336, 350, 364,

                196, 210, 224,
                336, 350, 364
            ).reshape(3, 2, 3),
            weight.grad()
        )

        assertTensorEquals(
            tensorOf(14).expand(3),
            bias.grad()
        )
    }

    @RepeatedTest(20)
    fun should_calculate_conv2d_pad0_dilation1_stride3_with_grads() {
        val t1 = tensorOf(
            1, 1, 1, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1,
            1, 0, 0, 1, 1, 0, 0, 1, 1,
            1, 0, 0, 1, 1, 0, 0, 1, 1,
            1, 0, 0, 1, 1, 0, 0, 1, 1,
            1, 1, 1, 0, 0, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 1, 1, 1,
            1, 0, 0, 0, 1, 0, 0, 1, 1,
            0, 0, 0, 0, 1, 0, 0, 0, 1,

            0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 1, 1, 0, 0,
            0, 1, 1, 0, 0, 1, 1, 0, 0,
            0, 1, 1, 0, 0, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 0, 0, 0,
            0, 1, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 0, 1, 1, 1, 0
        ).reshape(2, 1, 9, 9).asVariable(requiresGrad = true)

        val weight = tensorOf(
            1, 2, 1,
            2, 1, 2,
            1, 2, 1
        ).reshape(1, 1, 3, 3).expand(3, 1, 3, 3).asVariable(requiresGrad = true)

        val bias = tensorOf(2, 2, 2, requiresGrad = true)

        val res = t1.conv2d(weight, bias, stride = 3 to 3)

        assertTensorEquals(
            tensorOf(
                    12,  8, 14,
                     9,  9, 12,
                     7,  5, 10,

                    12,  8, 14,
                     9,  9, 12,
                     7,  5, 10,

                    12,  8, 14,
                     9,  9, 12,
                     7,  5, 10,


                     5,  9,  3,
                     8,  8,  5,
                    10, 12,  7,

                     5,  9,  3,
                     8,  8,  5,
                    10, 12,  7,

                     5,  9,  3,
                     8,  8,  5,
                    10, 12,  7
            ).reshape(2, 3, 3, 3),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                6, 3, 6, 6, 3, 6, 6, 3, 6,
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                6, 3, 6, 6, 3, 6, 6, 3, 6,
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                6, 3, 6, 6, 3, 6, 6, 3, 6,
                3, 6, 3, 3, 6, 3, 3, 6, 3,


                3, 6, 3, 3, 6, 3, 3, 6, 3,
                6, 3, 6, 6, 3, 6, 6, 3, 6,
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                6, 3, 6, 6, 3, 6, 6, 3, 6,
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                3, 6, 3, 3, 6, 3, 3, 6, 3,
                6, 3, 6, 6, 3, 6, 6, 3, 6,
                3, 6, 3, 3, 6, 3, 3, 6, 3
            ).reshape(2, 1, 9, 9),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(9).expand(3, 1, 3, 3),
            weight.grad()
        )

        assertTensorEquals(
            tensorOf(18).expand(3),
            bias.grad()
        )
    }

    private fun genericConvTest(testCasePath: String, op: (t1: Tensor, weight: Tensor, bias: Tensor?) -> Tensor) {
        val testCase = TestCase.loadTestCase(testCasePath)

        val t1 = testCase["t1"]!!.asVariable(requiresGrad = true)
        val weight = testCase["weight"]!!.asVariable(requiresGrad = true)
        val bias = testCase["bias"]?.asVariable(requiresGrad = true)

        val expectedRes = testCase["res"]!!
        val expectedT1Grad = testCase["expectedT1Grad"]!!
        val expectedWeightGrad = testCase["expectedWeightGrad"]!!
        val expectedBiasGrad = testCase["expectedBiasGrad"]

        val time = System.nanoTime()

        val res = op(t1, weight, bias)

        Tensor.sum(res).backward()

        val duration = Duration.ofNanos(System.nanoTime() - time)
        println("exec time: ${duration.toMillis()}ms")

        assertAll(
            { assertTensorEquals(expectedRes, res, 1e-4f) },
            { assertTensorEquals(expectedT1Grad, t1.grad(), 1e-4f) },
            { assertTensorEquals(expectedWeightGrad, weight.grad(), 1e-4f) },
            {
                if (bias != null && expectedBiasGrad != null)
                    assertTensorEquals(expectedBiasGrad, bias.grad(), 1e-4f)
            }
        )
    }

    @RepeatedTest(20)
    fun should_calculate_conv2dTranspose_with_grads_stride3() {
        genericConvTest("/conv2dtranspose_stride3.xml") { t1, weight, bias ->
            t1.conv2dTranspose(weight, bias, stride = 3 to 3)
        }
    }

    @RepeatedTest(20)
    fun should_calculate_conv2dTranspose_for_stride3_padding2_outpadding1_groups2_dilation2() {
        genericConvTest("/conv2dtranspose_s3_p2_op1_g2_d2.xml") { t1, weight, bias ->
            t1.conv2dTranspose(weight, bias, stride = 3 to 3, padding = 2 to 2, outputPadding = 1 to 1, dilation = 2 to 2, groups = 2)
        }
    }

}