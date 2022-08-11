package eu.redbean.kten.tensor.tests

import eu.redbean.kten.api.autograd.functions.nn.batchNorm
import eu.redbean.kten.api.autograd.functions.nn.softmaxCrossEntropy
import eu.redbean.kten.api.autograd.functions.nn.upsample2d
import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.*
import eu.redbean.kten.api.tensor.Constants.all
import eu.redbean.kten.api.tensor.Tensor.Companion.arrange
import eu.redbean.kten.api.tensor.Tensor.Companion.concat
import eu.redbean.kten.api.tensor.Tensor.Companion.log
import eu.redbean.kten.api.tensor.Tensor.Companion.mean
import eu.redbean.kten.api.tensor.Tensor.Companion.randomTensor
import eu.redbean.kten.api.tensor.Tensor.Companion.sqrt
import eu.redbean.kten.api.tensor.Tensor.Companion.sum
import eu.redbean.kten.api.tensor.Tensor.Companion.tensorOf
import eu.redbean.kten.api.tensor.operations.nn.UpsampleType
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.serialization.serializers.tensorFromBinary
import eu.redbean.kten.api.tensor.serialization.serializers.tensorFromJson
import eu.redbean.kten.api.tensor.serialization.serializers.toBinary
import eu.redbean.kten.api.tensor.serialization.serializers.toJson
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotSame
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertAll
import kotlin.random.Random

abstract class TensorTestsBase {

    @Test
    fun should_calculate_addition() {
        val t1 = Tensor.tensorOf(
            1, 2, 3, 4,
            5, 6, 7, 8,
        ).reshape(2, 4)

        val t2 = Tensor.tensorOf(
            1,
            2
        ).reshape(2, 1)

        val res = t1 + t2

        assertTensorEquals(
            tensorOf(
                2, 3, 4, 5,
                7, 8, 9, 10
            ).reshape(2, 4),
            res
        )
    }

    @Test
    fun should_calculate_mse_and_grads() {
        val t1 = Tensor.tensorOf(
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ).reshape(2, 3).asVariable(requiresGrad = true)
        val t2 = Tensor.tensorOf(
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9
        ).reshape(2, 3).asVariable(requiresGrad = true)

        val error = mean((t1 - t2) pow 2f)

        error.backward()

        assertTensorEquals(
            tensorOf(0.09),
            error
        )

        assertTensorEquals(
            tensorOf(-0.1).expand(2, 3),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(0.1).expand(2, 3),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_min_with_gradients() {
        val t1 = Tensor.tensorOf(
            0.1, 0.2, 0.1,
            0.2, 1.0, 0.2,
            0.1, 0.2, 0.1
        ).reshape(1, 1, 3, 3).asVariable(requiresGrad = true)

        val min = t1.max(axis = -1)

        assertTensorEquals(
            tensorOf(
                0.2,
                1.0,
                0.2
            ).reshape(1, 1, 3),
            min
        )

        min.sum().backward()

        assertTensorEquals(
            tensorOf(
                0, 1, 0,
                0, 1, 0,
                0, 1, 0
            ).reshape(1, 1, 3, 3),
            t1.grad()
        )
    }

    @Test
    fun should_set_element_from_tensor_and_calculate_gradients() {
        val t1 = Tensor.tensorOf(
            0.1, 0.1, 0.1,
            requiresGrad = true
        )

        val t2 = Tensor.tensorOf(
            1, 1, 1,
            1, 1, 1,
            1, 1, 1
        ).reshape(3, 3).asVariable(requiresGrad = true)

        t2[1, 1] = t1[1] pow 2

        t2.sum().backward()
        t1.sum().backward()

        assertTensorEquals(
            tensorOf(
                1f, 1f, 1f,
                1f, 0.01f, 1f,
                1f, 1f, 1f
            ).reshape(3, 3),
            t2
        )

        assertTensorEquals(
            tensorOf(
                1.0, 1.2, 1.0
            ),
            t1.grad()
        )
    }

    @Test
    @Suppress("EmptyRange")
    fun should_set_element_range_from_tensor_and_calculate_gradients() {
        val t1 = Tensor.tensorOf(0.1).expand(3, 3).asVariable(requiresGrad = true)
        val t2 = Tensor.tensorOf(1).expand(3, 3, 3).asVariable(requiresGrad = true)

        t2[1..1, all, 1..-1] = (t1[all, 1..-1] pow 2).unsqueeze(0)

        t2.sum().backward()
        t1.sum().backward()

        assertTensorEquals(
            tensorOf(
                1f, 1f, 1f,
                1f, 1f, 1f,
                1f, 1f, 1f,
                1f, 0.01f, 0.01f,
                1f, 0.01f, 0.01f,
                1f, 0.01f, 0.01f,
                1f, 1f, 1f,
                1f, 1f, 1f,
                1f, 1f, 1f
            ).reshape(3, 3, 3),
            t2
        )

        assertTensorEquals(
            tensorOf(
                1f, 1.2f, 1.2f,
                1f, 1.2f, 1.2f,
                1f, 1.2f, 1.2f,
            ).reshape(3, 3),
            t1.grad()
        )
    }

    @Test
    fun should_set_element_to_constant_at_single_index_and_calculate_grads() {
        val t1 = Tensor.tensorOf(0.1).expand(2, 4).asVariable(requiresGrad = true)

        t1[-1, -2] = 1f

        t1.sum().backward()

        assertTensorEquals(
            tensorOf(
                0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 1.0, 0.1
            ).reshape(2, 4),
            t1
        )

        assertTensorEquals(
            tensorOf(
                1, 1, 1, 1,
                1, 1, 0, 1
            ).reshape(2, 4),
            t1.grad()
        )
    }

    @Test
    fun should_set_elements_by_range_and_calculate_grads() {
        val t1 = Tensor.tensorOf(0.1).expand(2, 4).asVariable(requiresGrad = true)

        t1[-1..-1, 1..2] = 1f

        assertTensorEquals(
            tensorOf(
                0.1, 0.1, 0.1, 0.1,
                0.1, 1.0, 1.0, 0.1
            ).reshape(2, 4),
            t1
        )

        t1.sum().backward()

        assertTensorEquals(
            tensorOf(
                1, 1, 1, 1,
                1, 0, 0, 1
            ).reshape(2, 4),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_broadcasting_addition_with_grads() {
        val t1 = tensorOf(1).expand(3, 3).asVariable(requiresGrad = true)
        val t2 = tensorOf(1).expand(3).asVariable(requiresGrad = true)

        val summary = sum((t1 + t2) pow 2)

        assertEquals(36f, summary.item())

        summary.backward()

        assertTensorEquals(
            tensorOf(4).expand(3, 3),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(12).expand(3),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_broadcasting_minus_with_grads() {
        val t1 = tensorOf(3).expand(3, 3).asVariable(requiresGrad = true)
        val t2 = tensorOf(1, requiresGrad = true)

        val res = sum(t1 - t2)

        assertEquals(18f, res.item())

        res.backward()

        assertTensorEquals(
            tensorOf(1).expand(3, 3),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(-9),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_tensor_constant_addition_with_grad() {
        val t1 = tensorOf(
            1, 2, 3,
            4, 5, 6
        ).reshape(2, 3).asVariable(requiresGrad = true)

        val tcRes = t1 + 2f
        val ctRes = 2f + t1

        val expected = tensorOf(
            3, 4, 5,
            6, 7, 8
        ).reshape(2, 3)

        assertTensorEquals(expected, tcRes)
        assertTensorEquals(expected, ctRes)

        sum(tcRes).backward()

        assertTensorEquals(
            tensorOf(1).expand(2, 3),
            t1.grad()
        )

        (t1 as Variable).zeroGrad()

        sum(ctRes).backward()

        assertTensorEquals(
            tensorOf(1).expand(2, 3),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_tensor_constant_minus_with_grads() {
        val t1 = tensorOf(
            1, 2, 3,
            4, 5, 6
        ).reshape(2, 3).asVariable(requiresGrad = true)

        val tcRes = t1 - 2f
        val ctRes = 2f - t1

        val expected = tensorOf(
            -1, 0, 1,
            2, 3, 4
        ).reshape(2, 3)

        assertTensorEquals(expected, tcRes)
        assertTensorEquals(-expected, ctRes)

        sum(tcRes).backward()

        assertTensorEquals(
            tensorOf(1).expand(2, 3),
            t1.grad()
        )

        (t1 as Variable).zeroGrad()

        sum(ctRes).backward()

        assertTensorEquals(
            tensorOf(-1).expand(2, 3),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_tensor_times_tensor_with_grads() {
        val t1 = tensorOf(2).expand(3, 3).asVariable(requiresGrad = true)
        val t2 = tensorOf(3).expand(3).asVariable(requiresGrad = true)

        val res = sum(t1 * t2)

        assertEquals(54f, res.item())

        res.backward()

        assertTensorEquals(
            tensorOf(3).expand(3, 3),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(6).expand(3),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_tensor_div_tensor_with_grads() {
        val t1 = tensorOf(6).expand(3, 3).asVariable(requiresGrad = true)
        val t2 = tensorOf(2).expand(3, 1).asVariable(requiresGrad = true)

        val res = sum(t1 / t2)

        assertEquals(27f, res.item())

        res.backward()

        assertTensorEquals(
            tensorOf(0.5).expand(3, 3),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(-4.5).expand(3, 1),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_tensor_constant_times_with_grads() {
        val t1 = tensorOf(
            1, 2, 3, 4,
            5, 6, 7, 8
        ).reshape(2, 4).asVariable(requiresGrad = true)

        val tcRes = t1 * 2f
        val ctRes = 2f * t1

        val expected = tensorOf(
            2, 4, 6, 8,
            10, 12, 14, 16
        ).reshape(2, 4)

        assertTensorEquals(expected, tcRes)
        assertTensorEquals(expected, ctRes)

        sum(tcRes).backward()

        assertTensorEquals(tensorOf(2).expand(2, 4), t1.grad())

        (t1 as Variable).zeroGrad()

        sum(ctRes).backward()

        assertTensorEquals(tensorOf(2).expand(2, 4), t1.grad())
    }

    @Test
    fun should_calculate_tensor_constant_div_with_grads() {
        val t1 = tensorOf(
            1, 2, 3, 4,
            5, 6, 7, 8
        ).reshape(2, 4).asVariable(requiresGrad = true)

        val tcRes = t1 / 2f
        val ctRes = 2f / t1

        assertTensorEquals(
            tensorOf(
                0.5, 1.0, 1.5, 2.0,
                2.5, 3.0, 3.5, 4.0
            ).reshape(2, 4),
            tcRes
        )

        assertTensorEquals(
            tensorOf(
                2.0, 1.0, 0.6666667, 0.5,
                0.4, 0.3333334, 0.2857143, 0.25
            ).reshape(2, 4),
            ctRes
        )

        sum(tcRes).backward()

        assertTensorEquals(tensorOf(0.5).expand(2, 4), t1.grad())

        (t1 as Variable).zeroGrad()

        sum(ctRes).backward()

        assertTensorEquals(
            tensorOf(
                -2.0, -0.5, -0.22222224, -0.125,
                -0.080000006, -0.05555556, -0.04081633, -0.03125
            ).reshape(2, 4),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_tensor_pow_tensor_with_grads() {
        val t1 = tensorOf(
            2, 3, 2,
            3, 3, 3,
            2, 3, 2
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val t2 = tensorOf(2, 3, 2).reshape(1, 3).asVariable(requiresGrad = true)

        val res = sum(t1 pow t2)

        assertEquals(115f, res.item())

        res.backward()

        assertTensorEquals(
            tensorOf(
                4, 27, 4,
                6, 27, 6,
                4, 27, 4
            ).reshape(3, 3),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(15.432688, 88.987595, 15.432688).reshape(1, 3),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_constant_tensor_pow_with_grads() {
        val t1 = tensorOf(
            2, 3, 2,
            3, 3, 3,
            2, 3, 2
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = 2f pow t1

        assertTensorEquals(
            tensorOf(
                4, 8, 4,
                8, 8, 8,
                4, 8, 4
            ).reshape(3, 3),
            res
        )

        sum(res).backward()

        assertTensorEquals(
            tensorOf(
                2.7726, 5.5452, 2.7726,
                5.5452, 5.5452, 5.5452,
                2.7726, 5.5452, 2.7726
            ).reshape(3, 3),
            t1.grad(),
            1e-4f
        )
    }

    @Test
    fun should_calculate_sum_over_axis_with_grads() {
        val t1 = tensorOf(
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = t1.sum(axis = -1)

        assertTensorEquals(
            tensorOf(3, 6, 9),
            res
        )

        res.backward(tensorOf(2).expand(3))

        assertTensorEquals(
            tensorOf(2).expand(3, 3),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_sum_over_axis_keep_dims_with_grads() {
        val t1 = tensorOf(
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = t1.sum(axis = 0, keepDimensions = true)

        assertTensorEquals(
            tensorOf(6).expand(1, 3),
            res
        )

        res.backward(tensorOf(2).expand(1, 3))

        assertTensorEquals(
            tensorOf(2).expand(3, 3),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_mean_over_axis_with_grads() {
        val t1 = tensorOf(
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = t1.mean(axis = 0)

        assertTensorEquals(tensorOf(2).expand(3), res)

        res.backward(tensorOf(3).expand(3))

        assertTensorEquals(tensorOf(1).expand(3, 3), t1.grad())
    }

    @Test
    fun should_calculate_mean_over_axis_keep_dims_with_grads() {
        val t1 = tensorOf(
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = t1.mean(axis = -1, keepDimensions = true)

        assertTensorEquals(tensorOf(1, 2, 3).reshape(3, 1), res)

        res.backward(tensorOf(3).expand(3, 1))

        assertTensorEquals(tensorOf(1).expand(3, 3), t1.grad())
    }

    @Test
    fun should_calculate_max_over_axis_with_grads() {
        val t1 = tensorOf(
            3, 2, 1,
            1, 3, 2,
            3, 1, 2
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = t1.max(axis = -1)

        assertTensorEquals(tensorOf(3).expand(3), res)

        sum(res).backward()

        assertTensorEquals(
            tensorOf(
                1, 0, 0,
                0, 1, 0,
                1, 0, 0
            ).reshape(3, 3),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_max_over_axis_keep_dims_with_grads() { // TODO check against torch
        val t1 = tensorOf(
            3, 2, 1,
            1, 3, 2,
            3, 1, 2
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = t1.max(axis = 0, keepDimensions = true)

        assertTensorEquals(tensorOf(3, 3, 2).expand(1, 3), res)

        sum(res).backward()

        assertTensorEquals(
            tensorOf(
                1, 0, 0,
                0, 1, 1,
                0, 0, 0
            ).reshape(3, 3),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_min_over_axis_with_grads() {
        val t1 = tensorOf(
            3, 2, 1,
            1, 3, 2,
            3, 1, 2
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = t1.min(axis = -1)

        assertTensorEquals(tensorOf(1).expand(3), res)

        sum(res).backward()

        assertTensorEquals(
            tensorOf(
                0, 0, 1,
                1, 0, 0,
                0, 1, 0
            ).reshape(3, 3),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_min_over_axis_keep_dims_with_grads() {
        val t1 = tensorOf(
            3, 2, 1,
            1, 3, 2,
            3, 1, 2
        ).reshape(3, 3).asVariable(requiresGrad = true)

        val res = t1.min(axis = 0, keepDimensions = true)

        assertTensorEquals(tensorOf(1).expand(1, 3), res)

        sum(res).backward()

        assertTensorEquals(
            tensorOf(
                0, 0, 1,
                1, 0, 0,
                0, 1, 0
            ).reshape(3, 3),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_argmax() {
        val t1 = tensorOf(
            3, 2, 1,
            1, 3, 2,
            3, 1, 2
        ).reshape(3, 3)

        val res = t1.argMax(axis = -1)

        assertTensorEquals(tensorOf(0, 1, 0), res)
    }

    @Test
    fun should_calculate_argmax_keep_dims() {
        val t1 = tensorOf(
            3, 2, 1,
            1, 3, 2,
            3, 1, 2
        ).reshape(3, 3)

        val res = t1.argMax(axis = 0, keepDimensions = true)

        assertTensorEquals(tensorOf(0, 1, 1).reshape(1, 3), res)
    }

    @Test
    fun should_calculate_argmin() {
        val t1 = tensorOf(
            3, 2, 1,
            1, 3, 2,
            3, 1, 2
        ).reshape(3, 3)

        val res = t1.argMin(axis = -1)

        assertTensorEquals(tensorOf(2, 0, 1), res)
    }

    @Test
    fun should_calculate_argmin_keep_dims() {
        val t1 = tensorOf(
            3, 2, 1,
            1, 3, 2,
            3, 1, 2
        ).reshape(3, 3)

        val res = t1.argMin(axis = 0, keepDimensions = true)

        assertTensorEquals(tensorOf(1, 2, 0).reshape(1, 3), res)
    }

    @Test
    fun should_calculate_exp_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.exp(t1)

        assertTensorEquals(
            tensorOf(
                1.1052, 1.2214, 1.3499,
                1.4918, 1.6487, 1.8221
            ).reshape(2, 3),
            res,
            1e-4f
        )

        res.mean().backward()

        assertTensorEquals(
            tensorOf(
                0.1842, 0.2036, 0.2250,
                0.2486, 0.2748, 0.3037
            ).reshape(2, 3),
            t1.grad(),
            1e-4f
        )
    }

    private fun elementwiseTestTensor() = tensorOf(
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6
    ).reshape(2, 3).asVariable(requiresGrad = true)

    fun assertElementwiseOpResultAndGrads(t1: Tensor, res: Tensor, expectedRes: Tensor, expectedGrads: Tensor) {
        assertTensorEquals(expectedRes.reshape(2, 3), res, 1e-4f)
        res.sum().backward()
        assertTensorEquals(expectedGrads.reshape(2, 3), t1.grad(), 1e-4f)
    }

    @Test
    fun should_calculate_log_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.log(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                -2.3026, -1.6094, -1.2040,
                -0.9163, -0.6931, -0.5108
            ),
            tensorOf(
                10.0000, 5.0000, 3.3333,
                2.5000, 2.0000, 1.6667
            )
        )
    }

    @Test
    fun should_calculate_tanh_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.tanh(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.0997, 0.1974, 0.2913,
                0.3799, 0.4621, 0.5370
            ),
            tensorOf(
                0.9901, 0.9610, 0.9151,
                0.8556, 0.7864, 0.7116
            )
        )
    }

    @Test
    fun should_calculate_sigmoid_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.sigmoid(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.5250, 0.5498, 0.5744,
                0.5987, 0.6225, 0.6457
            ),
            tensorOf(
                0.2494, 0.2475, 0.2445,
                0.2403, 0.2350, 0.2288
            )
        )
    }

    @Test
    fun should_calculate_sinh_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.sinh(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.1002, 0.2013, 0.3045,
                0.4108, 0.5211, 0.6367
            ),
            tensorOf(
                1.0050, 1.0201, 1.0453,
                1.0811, 1.1276, 1.1855
            )
        )
    }

    @Test
    fun should_calculate_cosh_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.cosh(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                1.0050, 1.0201, 1.0453,
                1.0811, 1.1276, 1.1855
            ),
            tensorOf(
                0.1002, 0.2013, 0.3045,
                0.4108, 0.5211, 0.6367
            )
        )
    }

    @Test
    fun should_calculate_abs_with_grads() {
        var t1 = elementwiseTestTensor().noGrad()

        t1[1, 1] = -0.4f

        t1 = t1.asVariable(requiresGrad = true)

        val res = Tensor.abs(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.1, 0.2, 0.3,
                0.4, 0.4, 0.6
            ),
            tensorOf(
                1, 1, 1,
                1, -1, 1
            )
        )
    }

    @Test
    fun should_calculate_clamp_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = t1.clamp(0.15f, 0.55f)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.15, 0.2, 0.3,
                0.4, 0.5, 0.55
            ),
            tensorOf(
                0, 1, 1,
                1, 1, 0
            )
        )
    }

    @Test
    fun should_calculate_sqrt_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.sqrt(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.3162, 0.4472, 0.5477,
                0.6325, 0.7071, 0.7746
            ),
            tensorOf(
                1.5811, 1.1180, 0.9129,
                0.7906, 0.7071, 0.6455
            )
        )
    }

    @Test
    fun should_calculate_sin_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.sin(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.0998, 0.1987, 0.2955,
                0.3894, 0.4794, 0.5646
            ),
            tensorOf(
                0.9950, 0.9801, 0.9553,
                0.9211, 0.8776, 0.8253
            )
        )
    }

    @Test
    fun should_calculate_cos_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.cos(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.9950, 0.9801, 0.9553,
                0.9211, 0.8776, 0.8253
            ),
            tensorOf(
                -0.0998, -0.1987, -0.2955,
                -0.3894, -0.4794, -0.5646
            )
        )
    }

    @Test
    fun should_calculate_tan_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.tan(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.1003, 0.2027, 0.3093,
                0.4228, 0.5463, 0.6841
            ),
            tensorOf(
                1.0101, 1.0411, 1.0957,
                1.1788, 1.2984, 1.4680
            )
        )
    }

    @Test
    fun should_calculate_asin_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.asin(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.1002, 0.2014, 0.3047,
                0.4115, 0.5236, 0.6435
            ),
            tensorOf(
                1.0050, 1.0206, 1.0483,
                1.0911, 1.1547, 1.2500
            )
        )
    }

    @Test
    fun should_calculate_acos_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.acos(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                1.4706, 1.3694, 1.2661,
                1.1593, 1.0472, 0.9273
            ),
            tensorOf(
                -1.0050, -1.0206, -1.0483,
                -1.0911, -1.1547, -1.2500
            )
        )
    }

    @Test
    fun should_calculate_atan_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.atan(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                0.0997, 0.1974, 0.2915,
                0.3805, 0.4636, 0.5404
            ),
            tensorOf(
                0.9901, 0.9615, 0.9174,
                0.8621, 0.8000, 0.7353
            )
        )
    }

    @Test
    fun should_calculate_reciprocal_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = t1.reciprocal()

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                10.0000, 5.0000, 3.3333,
                2.5000, 2.0000, 1.6667
            ),
            tensorOf(
                -100.0000, -25.0000, -11.1111,
                -6.2500, -4.0000, -2.7778
            )
        )
    }

    @Test
    fun should_calculate_floor_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = t1.reciprocal().floor()

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                10, 5, 3,
                2, 2, 1
            ),
            tensorOf(0).expand(2, 3)
        )
    }

    @Test
    fun should_calculate_ceil_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = t1.reciprocal().ceil()

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                10, 5, 4,
                3, 2, 2
            ),
            tensorOf(0).expand(2, 3)
        )
    }

    @Test
    fun should_calculate_round_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = t1.reciprocal().round()

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                10, 5, 3,
                3, 2, 2
            ), //TODO check
            tensorOf(0).expand(2, 3)
        )
    }

    @Test
    fun should_calculate_trunc_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = t1.reciprocal().trunc()

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                10, 5, 3,
                2, 2, 1
            ),
            tensorOf(0).expand(2, 3)
        )
    }

    @Test
    fun should_calculate_sign_with_grads() {
        var t1 = elementwiseTestTensor().noGrad()

        t1[1, 1] = -0.4f

        t1 = t1.asVariable(requiresGrad = true)

        val res = Tensor.sign(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                1, 1, 1,
                1, -1, 1
            ),
            tensorOf(0).expand(2, 3)
        )
    }

    @Test
    fun should_calculate_rsqrt_with_grads() {
        val t1 = elementwiseTestTensor()

        val res = Tensor.rsqrt(t1)

        assertElementwiseOpResultAndGrads(
            t1, res,
            tensorOf(
                3.1623, 2.2361, 1.8257,
                1.5811, 1.4142, 1.2910
            ),
            tensorOf(
                -15.8114, -5.5902, -3.0429,
                -1.9764, -1.4142, -1.0758
            )
        )
    }

    @Test
    fun should_transpose_tensor_with_grads() {
        val t1 = tensorOf(
            1, 2, 3,
            4, 5, 6
        ).reshape(2, 3).asVariable(requiresGrad = true)

        val res = t1.transpose(0, 1)

        assertTensorEquals(
            tensorOf(
                1, 4,
                2, 5,
                3, 6
            ).reshape(3, 2),
            res
        ) //TODO check ocl

        res.backward(
            tensorOf(
                1, 0,
                0, 1,
                1, 0
            ).reshape(3, 2)
        )

        assertTensorEquals(
            tensorOf(
                1, 0, 1,
                0, 1, 0
            ).reshape(2, 3),
            t1.grad()
        )
    }

    @Test
    fun should_expand_tensor_with_grad() {
        val t1 = tensorOf(1, requiresGrad = true)

        t1.expand(3, 3).sum().backward()

        assertTensorEquals(tensorOf(9), t1.grad())
    }

    @Test
    fun should_permute_tensor_with_grads() {
        val t1 = tensorOf(
            1, 2,
            3, 4,
            5, 6,

            7, 8,
            9, 10,
            11, 12
        ).reshape(2, 3, 2).asVariable(requiresGrad = true)

        val res = t1.permute(2, 0, 1)

        assertTensorEquals(
            tensorOf(
                1, 3, 5,
                7, 9, 11,

                2, 4, 6,
                8, 10, 12
            ).reshape(2, 2, 3),
            res
        )

        sum(res pow 2f).backward()

        assertTensorEquals(
            tensorOf(
                2, 4,
                6, 8,
                10, 12,

                14, 16,
                18, 20,
                22, 24
            ).reshape(2, 3, 2),
            t1.grad()
        )
    }

    @Test
    fun should_concat_tensors_with_grads() {
        val t1 = tensorOf(1, 2, 3).expand(2, 3).asVariable(requiresGrad = true)
        val t2 = tensorOf(4, 5, 6).expand(1, 3).asVariable(requiresGrad = true)
        val t3 = tensorOf(7, 8, 9).expand(1, 3).asVariable(requiresGrad = true)

        val res = concat(t1, t2, t3)

        assertTensorEquals(
            tensorOf(
                1, 2, 3,
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ).reshape(4, 3),
            res
        )

        sum(res pow 2f).backward()

        assertTensorEquals(
            tensorOf(
                2, 4, 6,
                2, 4, 6
            ).reshape(2, 3),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(
                8, 10, 12
            ).reshape(1, 3),
            t2.grad()
        )

        assertTensorEquals(
            tensorOf(
                14, 16, 18
            ).reshape(1, 3),
            t3.grad()
        )
    }

    //TODO test squeeze, unsqeeze with grads (not really interesting)

    @Test
    fun should_calculate_gather_with_grads() {
        val t1 = tensorOf(
            1, 2,
            3, 4
        ).reshape(2, 2).asVariable(requiresGrad = true)

        val index = tensorOf(
            0, 0,
            1, 0
        ).reshape(2, 2)

        val res = t1.gather(1, index)

        assertTensorEquals(
            tensorOf(
                1, 1,
                4, 3
            ).reshape(2, 2),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                2, 0,
                1, 1
            ).reshape(2, 2),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_scatter() {
        val t1 = Tensor.arrange(11f, start = 1f).reshape(2, 5)

        val index1 = tensorOf(
            0, 1, 2, 0
        ).reshape(1, 4)

        val res1 = Tensor.zeros(3, 5).scatter(0, index1, t1)

        val index2 = tensorOf(
            0, 1, 2,
            0, 1, 4
        ).reshape(2, 3)

        val res2 = Tensor.zeros(3, 5).scatter(1, index2, t1)

        assertTensorEquals(
            tensorOf(
                1, 0, 0, 4, 0,
                0, 2, 0, 0, 0,
                0, 0, 3, 0, 0
            ).reshape(3, 5),
            res1
        )

        assertTensorEquals(
            tensorOf(
                1, 2, 3, 0, 0,
                6, 7, 0, 0, 8,
                0, 0, 0, 0, 0
            ).reshape(3, 5),
            res2
        )
    }

    @Test
    fun should_calculate_scatter_with_grads() {
        val t1 = Tensor.arrange(11f, start = 1f).reshape(2, 5).asVariable(requiresGrad = true)

        val index = tensorOf(
            0, 2, 1, 3, 4,
            4, 1, 3, 2, 0
        ).reshape(2, 5)

        val target = Tensor.zeros(3, 5).asVariable(requiresGrad = true)

        val res = ((target + 1f).scatter(1, index, t1) pow 2)

        assertTensorEquals(
            tensorOf(
                1, 9, 4, 16, 25,
                100, 49, 81, 64, 36,
                1, 1, 1, 1, 1
            ).reshape(3, 5),
            res, 1e-5f
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                2, 4, 6, 8, 10,
                12, 14, 16, 18, 20
            ).reshape(2, 5),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                2, 2, 2, 2, 2
            ).reshape(3, 5),
            target.grad()
        )
    }

    @Test
    fun should_index_select_with_gradients() {
        val tensor = arrange(27).reshape(3, 3, 3).asVariable(requiresGrad = true)
        val index = tensorOf(0, 2, 0)

        val res = tensor.indexSelect(1, index)

        assertTensorEquals(
            tensorOf(
                0, 1, 2,
                6, 7, 8,
                0, 1, 2,

                9, 10, 11,
                15, 16, 17,
                9, 10, 11,

                18, 19, 20,
                24, 25, 26,
                18, 19, 20
            ).reshape(3, 3, 3),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                2, 2, 2,
                0, 0, 0,
                1, 1, 1,

                2, 2, 2,
                0, 0, 0,
                1, 1, 1,

                2, 2, 2,
                0, 0, 0,
                1, 1, 1
            ).reshape(3, 3, 3),
            tensor.grad()
        )
    }

    @Test
    fun should_calculate_dot_with_grad() {
        val t1 = Tensor.arrange(10).reshape(2, 5).asVariable(requiresGrad = true)

        val res = t1[0].dot(t1[1])

        assertTensorEquals(
            tensorOf(80),
            res
        )

        res.backward()

        assertTensorEquals(
            tensorOf(
                5, 6, 7, 8, 9,
                0, 1, 2, 3, 4
            ).reshape(2, 5),
            t1.grad()
        )
    }

    @Test
    fun should_calculate_matrix_vector_multiplication_with_grads() {
        val t1 = Tensor.arrange(10).reshape(2, 5).asVariable(requiresGrad = true)
        val t2 = tensorOf(5, 6, 7, 8, 9, requiresGrad = true)

        val res = t1 matmul t2

        assertTensorEquals(
            tensorOf(80, 255),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                5, 6, 7, 8, 9,
                5, 6, 7, 8, 9
            ).reshape(2, 5),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(5, 7, 9, 11, 13),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_vector_matrix_multiplication_with_grads() {
        val t1 = Tensor.arrange(10).reshape(2, 5).asVariable(requiresGrad = true)
        val t2 = tensorOf(5, 6, requiresGrad = true)

        val res = t2 matmul t1
        //TODO check ocl
        assertTensorEquals(
            tensorOf(30, 41, 52, 63, 74),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                5, 5, 5, 5, 5,
                6, 6, 6, 6, 6
            ).reshape(2, 5),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(10, 35),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_matrix_matrix_multiplication_with_grads() {
        val t1 = Tensor.arrange(10).reshape(5, 2).asVariable(requiresGrad = true)
        val t2 = tensorOf(
            5, 6, 7,
            8, 9, 10
        ).reshape(2, 3).asVariable(requiresGrad = true)

        val res = t1 matmul t2
        //TODO check ocl
        assertTensorEquals(
            tensorOf(
                8, 9, 10,
                34, 39, 44,
                60, 69, 78,
                86, 99, 112,
                112, 129, 146
            ).reshape(5, 3),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                18, 27,
                18, 27,
                18, 27,
                18, 27,
                18, 27
            ).reshape(5, 2),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(
                20, 20, 20,
                25, 25, 25
            ).reshape(2, 3),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_matmul_reused_with_grads() {
        val t1 = Tensor.arrange(10).reshape(5, 2)
        val t2 = tensorOf(
            5, -6, 7,
            -8, 9, -10
        ).reshape(2, 3).asVariable(requiresGrad = true)
        val t3 = tensorOf(1, 2, 3, requiresGrad = true)

        val x = (t1 matmul t2) + t3

        var res = x.clamp(min = 0f) + 0.2f * x.clamp(max = 0f)
        res = res matmul (res.transpose(0, 1) / 1000f)
        res = sum(res + (res pow 2))

        assertTensorEquals(
            tensorOf(24.0433),
            res,
            1e-4f
        )

        res.backward()

        assertTensorEquals(
            tensorOf(
                -0.4020, 11.9788, -0.4020,
                -0.4895, 14.5885, -0.4895
            ).reshape(2, 3),
            t2.grad(),
            1e-4f
        )

        assertTensorEquals(
            tensorOf(-0.0875, 2.6097, -0.0875),
            t3.grad(),
            1e-4f
        )
    }

    @Test
    fun should_calculate_3d_tensor_matrix_multiplication_with_grads() {
        val t1 = Tensor.arrange(2 * 3 * 4).reshape(2, 3, 4).asVariable(requiresGrad = true)
        val t2 = Tensor.arrange(4 * 2).reshape(4, 2).asVariable(requiresGrad = true)

        val res = t1 matmul t2

        assertTensorEquals(
            tensorOf(
                28, 34,
                76, 98,
                124, 162,

                172, 226,
                220, 290,
                268, 354
            ).reshape(2, 3, 2),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                1, 5, 9, 13
            ).expand(2, 3, 4),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(
                60,
                66,
                72,
                78
            ).unsqueeze(-1).expand(4, 2),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_multiple_batch_matmul_with_grads() {
        val t1 = Tensor.arrange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).asVariable(requiresGrad = true)
        val t2 = Tensor.arrange(2 * 3 * 5 * 2).reshape(2, 3, 5, 2).asVariable(requiresGrad = true)

        val res = t1 matmul t2

        assertTensorEquals(
            tensorOf(
                60, 70,
                160, 195,
                260, 320,
                360, 445,

                1560, 1670,
                1910, 2045,
                2260, 2420,
                2610, 2795,

                5060, 5270,
                5660, 5895,
                6260, 6520,
                6860, 7145,

                10560, 10870,
                11410, 11745,
                12260, 12620,
                13110, 13495,

                18060, 18470,
                19160, 19595,
                20260, 20720,
                21360, 21845,

                27560, 28070,
                28910, 29445,
                30260, 30820,
                31610, 32195
            ).reshape(2, 3, 4, 2),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                1, 5, 9, 13, 17,
                21, 25, 29, 33, 37,
                41, 45, 49, 53, 57,

                61, 65, 69, 73, 77,
                81, 85, 89, 93, 97,
                101, 105, 109, 113, 117
            ).reshape(2, 3, 1, 5).expand(2, 3, 4, 5),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(
                30, 34, 38, 42, 46,
                110, 114, 118, 122, 126,
                190, 194, 198, 202, 206,

                270, 274, 278, 282, 286,
                350, 354, 358, 362, 366,
                430, 434, 438, 442, 446
            ).reshape(2, 3, 5, 1).expand(2, 3, 5, 2),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_matrix_3d_tensor_multiplication_with_grads() {
        val t1 = Tensor.arrange(2 * 3).reshape(2, 3).asVariable(requiresGrad = true)
        val t2 = Tensor.arrange(2 * 3 * 2).reshape(2, 3, 2).asVariable(requiresGrad = true)

        val res = t1.matmul(t2)

        assertTensorEquals(
            tensorOf(
                10, 13,
                28, 40,

                28, 31,
                100, 112
            ).reshape(2, 2, 2),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                14, 22, 30
            ).expand(2, 3),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(
                3, 5, 7
            ).unsqueeze(-1).expand(2, 3, 2),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_3d_tensor_vector_multiplication_with_grads() {
        val t1 = Tensor.arrange(2 * 3 * 2).reshape(2, 3, 2).asVariable(requiresGrad = true)
        val t2 = Tensor.arrange(2).asVariable(requiresGrad = true)

        val res = t1 matmul t2

        assertTensorEquals(
            tensorOf(
                1, 3, 5,
                7, 9, 11
            ).reshape(2, 3),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                0, 1
            ).expand(2, 3, 2),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(30, 36),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_vector_3d_tensor_multiplication_with_grads() {
        val t1 = Tensor.arrange(2 * 3 * 2).reshape(2, 3, 2).asVariable(requiresGrad = true)
        val t2 = Tensor.arrange(3).asVariable(requiresGrad = true)

        val res = t2 matmul t1

        assertTensorEquals(
            tensorOf(
                10, 13,
                28, 31
            ).reshape(2, 2),
            res
        )

        res.sum().backward()

        assertTensorEquals(
            tensorOf(
                0, 1, 2
            ).unsqueeze(-1).expand(2, 3, 2),
            t1.grad()
        )

        assertTensorEquals(
            tensorOf(14, 22, 30),
            t2.grad()
        )
    }

    @Test
    fun should_calculate_long_graph_output_reuse_with_grads() {
        val t1 = Tensor.ones(5, 5).asVariable(requiresGrad = true)
        val t2 = Tensor.ones(5, 5).asVariable(requiresGrad = true)

        var a = t1 + 2f
        var b = t2 + 2f
        var c = t1 + 2f

        val add1 = a + b
        val add2 = add1 + c

        for (i in 0 until 4) {
            a *= 2f
            b *= 2f
            c *= 2f
        }

        val branch = a + b + c

        val out = add2 + branch

        out.backward(Tensor.ones(5, 5))

        assertTensorEquals(
            Tensor.ones(5, 5) * 34f,
            t1.grad()
        )

        assertTensorEquals(
            Tensor.ones(5, 5) * 17f,
            t2.grad()
        )
    }

    @Test
    fun should_fill_tensor_partialy() {
        var t1 = Tensor.ones(2, 3)
        t1 *= 2f
        t1[0..0, 1..2] = 0f

        assertTensorEquals(
            tensorOf(
                2, 0, 0,
                2, 2, 2
            ).view(2, 3),
            t1
        )
    }

    //TODO test compare functions

    @Test
    fun should_serialize_deserialize_tensor_to_binary() {
        val t1 = randomTensor(2, 3, 4, 5, requiresGrad = true)
        sum(t1 pow 2).backward() //to have gradient values
        val t1NoGrad = t1.noGrad()

        val t1Bin = t1.toBinary()
        val t1NoGradBin = t1NoGrad.toBinary()

        val desT1 = Tensor.tensorFromBinary(t1Bin)
        val desT1NoGrad = Tensor.tensorFromBinary(t1NoGradBin)

        assertAll(
            { assertNotSame(t1, desT1) },
            { assertNotSame(t1NoGrad, desT1NoGrad) },
            { assertTensorEquals(t1, desT1) },
            { assertTensorEquals(t1NoGrad, desT1NoGrad) },
            { assertTensorEquals(t1.grad(), desT1.grad()) }
        )
    }

    @Test
    fun should_serialize_deserialize_tensor_to_json() {
        val t1 = randomTensor(2, 3, 4, 5, requiresGrad = true)
        sum(t1 pow 2).backward() //to have gradient values
        val t1NoGrad = t1.noGrad()

        val t1Json = t1.toJson()
        val t1NoGradJson = t1NoGrad.toJson()

        val desT1 = Tensor.tensorFromJson(t1Json)
        val desT1NoGrad = Tensor.tensorFromJson(t1NoGradJson)

        assertAll(
            { assertNotSame(t1, desT1) },
            { assertNotSame(t1NoGrad, desT1NoGrad) },
            { assertTensorEquals(t1, desT1) },
            { assertTensorEquals(t1NoGrad, desT1NoGrad) },
            { assertTensorEquals(t1.grad(), desT1.grad()) }
        )
    }

    @Test
    fun should_deserialize_tensor_from_json_string_not_created_by_serializer() {
        val t1String = """
            {
                "data" : [1., 2, -3., 4e-5, 5e3]
            }
        """.trimIndent()
        val t2String = """
            {
                "gradient": [[1,2,3], 
                             [4., 5., 6.1], 
                             [-7, -8, -0.9]],
                "data" : [[123, 123, 123], 
                          [4.56, 5.6, 6.0], 
                          [1e7, 1e8, 1e9]]
            }
        """.trimIndent()

        val t1 = Tensor.tensorFromJson(t1String)
        val t2 = Tensor.tensorFromJson(t2String)

        assertTensorEquals(
            tensorOf(1f, 2f, -3f, 4e-5f, 5e3f),
            t1
        )

        assertTensorEquals(
            tensorOf(
                1f, 2f, 3f,
                4f, 5f, 6.1f,
                -7f, -8f, -0.9f
            ).reshape(3, 3),
            t2.grad()
        )

        assertTensorEquals(
            tensorOf(
                123f, 123f, 123f,
                4.56f, 5.6f, 6.0f,
                1e7f, 1e8f, 1e9f
            ).reshape(3, 3),
            t2
        )
    }

    @Test
    fun should_calculate_gradients_with_gradient_aggregator() {
        val t1 = (Tensor.arrange(4 * 3 * 5 * 6) * 0.2f).reshape(4, 3, 5, 6).asVariable(requiresGrad = true)
        val t2 = (Tensor.arrange(4 * 3 * 5 * 6) * 0.1f).reshape(4, 3, 6, 5).asVariable(requiresGrad = true)

        var input = t1 matmul t2
        input *= t1.sum() + t2.mean()
        input /= t2.sum(axis = -2, keepDimensions = true)

        val calc = (input as AGTensor).gradientAggregate {
            (it.mean(1, keepDimensions = true) pow 2f) + (it pow -3f)
        }

        val res = calc.sum()

        val t1True = (Tensor.arrange(4 * 3 * 5 * 6) * 0.2f).reshape(4, 3, 5, 6).asVariable(requiresGrad = true)
        val t2True = (Tensor.arrange(4 * 3 * 5 * 6) * 0.1f).reshape(4, 3, 6, 5).asVariable(requiresGrad = true)

        var inputTrue = t1True matmul t2True
        inputTrue *= t1True.sum() + t2True.mean()
        inputTrue /= t2True.sum(axis = -2, keepDimensions = true)

        val calcTrue = (inputTrue.mean(1, keepDimensions = true) pow 2f) + (inputTrue pow -3f)

        val resTrue = calcTrue.sum()

        assertTensorEquals(resTrue, res, 1e-4f)

        res.backward()
        resTrue.backward()

        assertTensorEquals(t1True.grad(), t1.grad(), 1e-4f)
        assertTensorEquals(t2True.grad(), t2.grad(), 1e-4f)
    }

    @Test
    fun should_calculate_batch_norm_for_specified_axis_in_train_mode() {
        val tensor = randomTensor(10, 3, 4, 5, requiresGrad = true)

        val axis = 1
        val momentum = 0.1f
        val epsilon = 1e-3f

        val gamma = randomTensor(1, 3, 1, 1, requiresGrad = true)
        val beta = randomTensor(1, 3, 1, 1, requiresGrad = true)

        val runningMeanBase = randomTensor(1, 3, 1, 1)
        val runningVarBase = randomTensor(1, 3, 1, 1)

        val t = tensor.permute(1, 0, 2, 3).view(3, -1)
        val (m, v) = t.meanVariance(1)
        val mean = m.view(1, 3, 1, 1)
        val variance = v.view(1, 3, 1, 1)

        val runningMean = (mean.noGrad() * momentum) + runningMeanBase * (1f - momentum)
        val runningVar = (variance.noGrad() * momentum) + runningVarBase * (1f - momentum)

        var result = (tensor - mean) / sqrt(variance + epsilon)

        result = result * gamma + beta

        result.sum().backward()

        val tensorGrad = tensor.grad()
        val gammaGrad = gamma.grad()
        val betaGrad = beta.grad()

        (tensor as Variable).zeroGrad()
        (gamma as Variable).zeroGrad()
        (beta as Variable).zeroGrad()

        val runningMean2 = runningMeanBase.reshape(3)
        val runningVar2 = runningVarBase.reshape(3)

        val result2 = tensor.batchNorm(axis, runningMean2, runningVar2, true, momentum, epsilon, gamma.view(3), beta.view(3))

        result2.sum().backward()

        assertTensorEquals(result, result2, 0.02f)
        assertTensorEquals(runningMean.view(3), runningMean2)
        assertTensorEquals(runningVar.view(3), runningVar2)
        assertTensorEquals(tensorGrad, tensor.grad())
        assertTensorEquals(gammaGrad, gamma.grad())
        assertTensorEquals(betaGrad, beta.grad())
    }

    @Test
    fun should_calculate_batch_norm_for_specified_axis_in_inference_mode() {
        val tensor = randomTensor(10, 3, 4, 5)

        val axis = 2
        val momentum = 0.1f
        val epsilon = 1e-3f

        val gamma = randomTensor(1, 1, 4, 1)
        val beta = randomTensor(1, 1, 4, 1)

        val runningMean = randomTensor(1, 1, 4, 1)
        val runningVar = randomTensor(1, 1, 4, 1)

        var result = (tensor - runningMean) / sqrt(runningVar + epsilon)

        result = result * gamma + beta

        val runningMean2 = runningMean.reshape(4)
        val runningVar2 = runningVar.reshape(4)

        val result2 = tensor.batchNorm(axis, runningMean2, runningVar2, false, momentum, epsilon, gamma.view(4), beta.view(4))

        assertTensorEquals(result, result2)
    }

    @Test
    fun should_calculate_batch_norm_in_training_mode_without_gamma_and_beta() {
        val tensor = randomTensor(10, 3, 4, 5, requiresGrad = true)

        val axis = 1
        val momentum = 0.1f
        val epsilon = 1e-3f

        val runningMeanBase = randomTensor(1, 3, 1, 1)
        val runningVarBase = randomTensor(1, 3, 1, 1)

        val t = tensor.permute(1, 0, 2, 3).view(3, -1)
        val (m, v) = t.meanVariance(1)
        val mean = m.view(1, 3, 1, 1)
        val variance = v.view(1, 3, 1, 1)

        val runningMean = (mean.noGrad() * momentum) + runningMeanBase * (1f - momentum)
        val runningVar = (variance.noGrad() * momentum) + runningVarBase * (1f - momentum)

        val result = (tensor - mean) / sqrt(variance + epsilon)

        result.sum().backward()

        val tensorGrad = tensor.grad()

        (tensor as Variable).zeroGrad()

        val runningMean2 = runningMeanBase.reshape(3)
        val runningVar2 = runningVarBase.reshape(3)

        val result2 = tensor.batchNorm(axis, runningMean2, runningVar2, true, momentum, epsilon, null, null)

        result2.sum().backward()

        assertTensorEquals(result, result2, 0.01f)
        assertTensorEquals(runningMean.view(3), runningMean2)
        assertTensorEquals(runningVar.view(3), runningVar2)
        assertTensorEquals(tensorGrad, tensor.grad())
    }

    @Test
    fun should_calculate_batchnorm_in_inference_mode_without_gamma_and_beta() {
        val tensor = randomTensor(10, 3, 4, 5)

        val axis = 2
        val momentum = 0.1f
        val epsilon = 1e-3f

        val runningMean = randomTensor(1, 1, 4, 1)
        val runningVar = randomTensor(1, 1, 4, 1)

        val result = (tensor - runningMean) / sqrt(runningVar + epsilon)

        val runningMean2 = runningMean.reshape(4)
        val runningVar2 = runningVar.reshape(4)

        val result2 = tensor.batchNorm(axis, runningMean2, runningVar2, false, momentum, epsilon, null, null)

        assertTensorEquals(result, result2)
    }

    @Test
    fun should_calculate_upsample_nearest_with_grads() {
        val tensor = Tensor.randomTensor(5, 3, 4, 5, requiresGrad = true)

        val scale = 3

        var upsWithGrad = tensor.view(5, 3, 4, 1, 5, 1)
            .expand(5, 3, 4, scale, 5, scale)
            .view(5, 3, scale * 4, scale * 5)

        val upsNoGrad = upsWithGrad.noGrad()
        upsWithGrad = Tensor.exp(upsWithGrad)
        upsWithGrad.sum().backward()
        val tensorGrad = tensor.grad()

        (tensor as Variable).zeroGrad()

        var ups2WithGrad = tensor.upsample2d(UpsampleType.NEAREST, scale)
        val ups2NoGrad = ups2WithGrad.noGrad()
        ups2WithGrad = Tensor.exp(ups2WithGrad)
        ups2WithGrad.sum().backward()

        assertTensorEquals(upsNoGrad, ups2NoGrad)
        assertTensorEquals(tensorGrad, tensor.grad())
    }

    @Test
    fun should_calculate_upsample_nearest_on_3d_tensor_with_grad() {
        val tensor = Tensor.randomTensor(3, 6, 5, requiresGrad = true)

        val scale = 4

        var upsWithGrad = tensor.view(3, 6, 1, 5, 1)
            .expand(3, 6, scale, 5, scale)
            .view(3, scale * 6, scale * 5)

        val upsNoGrad = upsWithGrad.noGrad()
        upsWithGrad = Tensor.exp(upsWithGrad)
        upsWithGrad.sum().backward()
        val tensorGrad = tensor.grad()

        (tensor as Variable).zeroGrad()

        var ups2WithGrad = tensor.upsample2d(UpsampleType.NEAREST, scale)
        val ups2NoGrad = ups2WithGrad.noGrad()
        ups2WithGrad = Tensor.exp(ups2WithGrad)
        ups2WithGrad.sum().backward()

        assertTensorEquals(upsNoGrad, ups2NoGrad)
        assertTensorEquals(tensorGrad, tensor.grad())
    }

    @Test
    fun should_calculate_softmax_cross_entropy_from_logits() {
        val tensor = randomTensor(256, 2048, requiresGrad = true)
        val tensor2 = tensor.copy().asVariable(requiresGrad = true)

        val targets = Tensor(256) { Random.nextInt(2048).toFloat() }
        val targetsOneHot = Tensor.oneHot(targets, 2048)

        val res = tensor.softmaxCrossEntropy(targets)
        res.backward()

        val expVal = Tensor.exp(tensor2 - tensor2.max(-1, keepDimensions = true))
        val softmax = expVal / expVal.sum(-1, keepDimensions = true)

        val softmaxNorm = (softmax / softmax.sum(-1, keepDimensions = true)).clamp(PlatformProvider.epsilon, 1f - PlatformProvider.epsilon)
        val res2 = mean(-(targetsOneHot * log(softmaxNorm)).sum(-1))
        res2.backward()

        assertTensorEquals(res2, res)
        assertTensorEquals(tensor2.grad(), tensor.grad())
    }

}