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
        return listOf(mayUnexpand(gradient.copy(shallow = true), aShape, ops), mayUnexpand(gradient.copy(shallow = true), bShape, ops))
    }
}

class Minus(
    ops: TensorOperations<AbstractRawTensor<Any>>
): BiTensorFunction(ops, true) {

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        output = a - b
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>> {
        return listOf(mayUnexpand(gradient.copy(shallow = true), aShape, ops), mayUnexpand(-gradient, bShape, ops))
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
        return listOf(mayUnexpand(gradient * b, aShape, ops), mayUnexpand(gradient * a, bShape, ops))
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
        val aGrad = gradient * bRec
        val bGrad = -gradient
        if (a.shape == b.shape) {
            bGrad *= a
            bGrad *= bRec
            bGrad *= bRec
        } else {
            bGrad *= a * bRec * bRec
        }
        ops.release(bRec)
        return listOf(
            mayUnexpand(aGrad, aShape, ops),
            mayUnexpand(bGrad, bShape, ops)
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

        val aGrad = gradient * b
        aGrad *= (a pow (b - 1f))

        val bGrad = gradient * (a pow b)
        bGrad *= a.log()

        return listOf(
            mayUnexpand(aGrad, aShape, ops),
            mayUnexpand(bGrad, bShape, ops)
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
        return listOf(gradient.copy(shallow = true))
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
            listOf(gradient.copy(shallow = true))
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

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        if (tensorFirst)
            return listOf(gradient / constant)
        else {
            val res = valuesSaved[0].reciprocal()
            res *= res
            res *= gradient
            res *= -constant
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
            val inputGrad = gradient * constant
            inputGrad *= input pow constant - 1f
            return listOf(inputGrad)
        } else {
            val (res) = valuesSaved
            val inputGrad = gradient * res
            inputGrad *= ln(constant)
            return listOf(inputGrad)
        }
    }
}


