package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor


class Exp(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        val res = input.exp()
        saveForBackward(res)
        this.output = res
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (res) = valuesSaved
        return listOf(gradient * res)
    }
}

class Log(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.log()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        return listOf(gradient / input)
    }
}

class Tanh(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        val res = input.tanh()
        saveForBackward(res)
        output = res
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (res) = valuesSaved

        val gradTanh = res * res
        gradTanh *= -1.0f
        gradTanh += 1.0f // mem optimized version of 1 - tanh(x)^2

        gradTanh *= gradient

        return listOf(gradTanh)
    }
}

class Sigmoid(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        val res = input.sigmoid()
        saveForBackward(res)
        output = res
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (res) = valuesSaved

        val gradInput = -res
        gradInput += 1.0f
        gradInput *= res
        gradInput *= gradient

        return listOf(gradInput)
    }
}

class Sinh(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.sinh()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        val gradSinh = input.cosh()
        gradSinh *= gradient
        return listOf(gradSinh)
    }
}

class Cosh(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.cosh()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        val gradCosh = input.sinh()
        gradCosh *= gradient
        return listOf(gradCosh)
    }
}

class Abs(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.abs()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        val gradInput = input.sign()
        gradInput *= gradient
        return listOf(gradInput)
    }
}

class Clamp(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    private var min: Float = 0.0f
    private var max: Float = 0.0f

    operator fun invoke(tensor: Tensor, min: Float, max: Float): Clamp {
        invoke(tensor)
        this.min = min
        this.max = max
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.clamp(min, max)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        val mask = (input gte min)
        val maxMask = input lte max
        mask *= maxMask
        val gradIn = gradient * mask
        ops.release(mask, maxMask)
        return listOf(gradIn)
    }
}

class Sqrt(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.sqrt()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        val gradInput = input pow -0.5f
        gradInput *= gradient
        gradInput /= 2.0f
        return listOf(gradInput)
    }
}

class Sin(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.sin()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        val gradInput = input.cos()
        gradInput *= gradient
        return listOf(gradInput)
    }
}

class Cos(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops){

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.cos()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        val gradInput = input.sin()
        gradInput *= gradient
        gradInput *= -1.0f
        return listOf(gradInput)
    }
}

class Tan(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.tan()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        val inputCos = input.cos()
        val inputCosSqr = inputCos pow 2.0f
        val gradInput = gradient / inputCosSqr
        ops.release(inputCos, inputCosSqr)
        return listOf(gradInput)
    }
}

class Asin(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.asin()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        return listOf(gradient * (-(input * input) + 1.0f).sqrt().reciprocal()) //TODO optimize
    }
}

class Acos(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.acos()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        return listOf(gradient * -((-(input * input) + 1.0f).sqrt().reciprocal())) // TODO optimize
    }
}

class Atan(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        saveForBackward(input)
        output = input.atan()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input) = valuesSaved
        return listOf(gradient * (input * input + 1.0f).reciprocal()) // TODO optimize
    }
}

class Reciprocal(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        val res = input.reciprocal()
        saveForBackward(res)
        output = res
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (res) = valuesSaved
        val resSqr = res * res
        val gradReciprocal = -resSqr
        val gradInput = gradient * gradReciprocal
        ops.release(resSqr, gradReciprocal)
        return listOf(gradInput)
    }
}

abstract class ConstantGrad(
    ops: TensorOperations<AbstractRawTensor<Any>>,
    val rawTensorFunction: (AbstractRawTensor<Any>) -> AbstractRawTensor<Any>,
    val gradientValue: Float = 0.0f,
): UnaryTensorFunction(ops) {

    lateinit var inputShape: List<Int>

    override fun doForward(input: AbstractRawTensor<Any>) {
        inputShape = input.shape
        output = rawTensorFunction(input)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(mayUnexpand(gradient * gradientValue, inputShape, ops))
    }
}

class Floor(
    ops: TensorOperations<AbstractRawTensor<Any>>
): ConstantGrad(ops, AbstractRawTensor<Any>::floor)

class Ceil(
    ops: TensorOperations<AbstractRawTensor<Any>>
): ConstantGrad(ops, AbstractRawTensor<Any>::ceil)

class Round(
    ops: TensorOperations<AbstractRawTensor<Any>>
): ConstantGrad(ops, AbstractRawTensor<Any>::round)

class Sign(
    ops: TensorOperations<AbstractRawTensor<Any>>
): ConstantGrad(ops, AbstractRawTensor<Any>::sign)

class Trunc(
    ops: TensorOperations<AbstractRawTensor<Any>>
): ConstantGrad(ops, AbstractRawTensor<Any>::trunc)

//TODO add frac, fmod, remainder (parameterization is needed for them, but otherwise constant grad functions with gradValue = 1)

class Rsqrt(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        val res = input.rsqrt()
        saveForBackward(res)
        output = res
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (res) = valuesSaved
        val gradRsqrt = res pow 3.0f
        gradRsqrt /= -2.0f
        val gradInput = gradient * gradRsqrt
        ops.release(gradRsqrt)
        return listOf(gradInput)
    }
}
