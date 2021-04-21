package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import kotlin.math.ln

class Plus(
    ops: TensorOperations<AbstractRawTensor<Any>>
): BiTensorFunction(ops, true) {

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        output = a + b
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>> {
        return listOf(mayUnexpand(gradient, aShape), mayUnexpand(gradient, bShape)) 
    }
}

class Minus(
    ops: TensorOperations<AbstractRawTensor<Any>>
): BiTensorFunction(ops, true) {

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        output = a - b
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>> {
        return listOf(mayUnexpand(gradient, aShape), mayUnexpand(-gradient, bShape))
    }
}

class Times(
    ops: TensorOperations<AbstractRawTensor<Any>>
): BiTensorFunction(ops, true) {

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        saveForBackward(a, b)
        output = a * b
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>> {
        val (a, b) = valuesSaved
        return listOf(mayUnexpand(gradient * b, aShape), mayUnexpand(gradient * a, bShape))
    }
}

class Div(
    ops: TensorOperations<AbstractRawTensor<Any>>
): BiTensorFunction(ops, true) {

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        saveForBackward(a, b)
        output = a / b
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>> {
        val (a, b) = valuesSaved
        val bRec = b.reciprocal()
        return listOf(
            mayUnexpand(gradient * bRec, aShape),
            mayUnexpand(-gradient * a * bRec * bRec, bShape)
        )
    }
}

class Pow(
    ops: TensorOperations<AbstractRawTensor<Any>>
): BiTensorFunction(ops, true) {

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        saveForBackward(a, b)
        output = a pow b
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>> {
        val (a, b) = valuesSaved
        return listOf(
            mayUnexpand(gradient * b * (a pow (b - 1.0f)), aShape),
            mayUnexpand(gradient * (a pow b) * a.log(), bShape)
        )
    }
}

class PlusConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): TensorConstantFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = input + constant
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(gradient)
    }
}

class MinusConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): TensorConstantFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        if (tensorFirst)
            output = input - constant
        else {
            output = -input
            if (constant != 0.0f)
                output!! += constant
        }
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return if (tensorFirst)
            listOf(gradient)
        else
            listOf(-gradient)
    }
}

class TimesConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): TensorConstantFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = input * constant
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(gradient * constant)
    }
}

class DivConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): TensorConstantFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        if (tensorFirst)
            output = input / constant
        else {
            saveForBackward(input)
            output = input.reciprocal()
            output!! *= constant
        }
    }

    /**
     * Note:
     * Since we cannot perform tensor - tensor multiplication inplace because of the possible shape changes,
     * so the calculation is broken up to enable us to manually release the references.
     */
    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        if (tensorFirst)
            return listOf(gradient / constant)
        else {
            val inputRec = valuesSaved[0].reciprocal()
            val inputRecSqr = inputRec * inputRec
            val res = gradient * inputRecSqr
            res *= -constant
            ops.release(inputRec, inputRecSqr)
            return listOf(res)
        }
    }

}

class PowConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): TensorConstantFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        if (tensorFirst) {
            saveForBackward(input)
            output = input pow constant
        } else {
            val res = ops.pow(constant, input)
            saveForBackward(res)
            output = res
        }
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        if (tensorFirst) {
            val (input) = valuesSaved
            val gradScaled = gradient * constant
            val inputPowCMinusOne = input pow constant - 1f
            val inputGrad = gradScaled * inputPowCMinusOne
            ops.release(gradScaled, inputPowCMinusOne)
            return listOf(inputGrad)
        } else {
            val (res) = valuesSaved
            val inputGrad = gradient * res
            inputGrad *= ln(constant)
            return listOf(inputGrad)
        }
    }
}


