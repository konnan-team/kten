package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.autograd.utils.dotShape
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

class GeneralMatrixMultiplication(
    ops: TensorOperations<AbstractRawTensor<Any>>
): AbstractBlasFunction(ops) {

    override fun doForward(addTensor: AbstractRawTensor<Any>, tensor1: AbstractRawTensor<Any>, tensor2: AbstractRawTensor<Any>) {
        saveForBackward(tensor1, tensor2)
        output = ops.gemm(addTensor, tensor1, tensor2, alpha, beta) // TODO size checks + shape calculations
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (tensor1, tensor2) = valuesSaved
        val addTensorGrad = calculateAddTensorGrad(gradient)
        var tensor1Grad: AbstractRawTensor<Any>? = null
        var tensor2Grad: AbstractRawTensor<Any>? = null

        if (inputs.second is AGTensor && inputs.second.requiresGrad) {
            tensor1Grad = ops.mm(gradient, tensor2, transposeSecond = true)
            if (alpha != 1f)
                tensor1Grad.timesAssign(alpha) // cannot use *= because of var
        }

        if (inputs.third is AGTensor && inputs.third.requiresGrad) {
            tensor2Grad = ops.mm(tensor1, gradient, transposeFirst = true)
            if (alpha != 1f)
                tensor2Grad.timesAssign(alpha)
        }

        return listOf(addTensorGrad, tensor1Grad, tensor2Grad)
    }
}

class BatchedGeneralMatrixMultiplication(
    ops: TensorOperations<AbstractRawTensor<Any>>
): AbstractBlasFunction(ops) {

    override fun doForward(addTensor: AbstractRawTensor<Any>, tensor1: AbstractRawTensor<Any>, tensor2: AbstractRawTensor<Any>) {
        saveForBackward(tensor1, tensor2)
        output = ops.gemmBatched(addTensor, tensor1, tensor2, alpha, beta)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> { // TODO consolidate with GEMM backward because it is the same, but with bmm
        val (tensor1, tensor2) = valuesSaved
        val addTensorGrad = calculateAddTensorGrad(gradient)
        var tensor1Grad: AbstractRawTensor<Any>? = null
        var tensor2Grad: AbstractRawTensor<Any>? = null

        if (inputs.second is AGTensor && inputs.second.requiresGrad) {
            tensor1Grad = ops.bmm(gradient, tensor2, transposeSecond = true)
            if (alpha != 1f)
                tensor1Grad.timesAssign(alpha)
        }

        if (inputs.third is AGTensor && inputs.third.requiresGrad) {
            tensor2Grad = ops.bmm(tensor1, gradient, transposeFirst = true)
            if (alpha != 1f)
                tensor2Grad.timesAssign(alpha)
        }

        return listOf(addTensorGrad, tensor1Grad, tensor2Grad)
    }
}

class GeneralMatrixVectorMultiplication(
    ops: TensorOperations<AbstractRawTensor<Any>>
): AbstractBlasFunction(ops) {

    override fun doForward(addTensor: AbstractRawTensor<Any>, tensor1: AbstractRawTensor<Any>, tensor2: AbstractRawTensor<Any>) {
        saveForBackward(tensor1, tensor2)
        output = ops.gemv(addTensor, tensor1, tensor2, alpha, beta)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (tensor1, tensor2) = valuesSaved
        val addTensorGrad = calculateAddTensorGrad(gradient)
        var tensor1Grad: AbstractRawTensor<Any>? = null
        var tensor2Grad: AbstractRawTensor<Any>? = null

        if (inputs.second is AGTensor && inputs.second.requiresGrad) {
            tensor1Grad = ops.outer(gradient, tensor2)
            if (alpha != 1f)
                tensor1Grad.timesAssign(alpha)
        }

        if (inputs.third is AGTensor && inputs.third.requiresGrad) {
            tensor2Grad = ops.mv(tensor1, gradient, transposeMatrix = true)
            if (alpha != 1f)
                tensor2Grad.timesAssign(alpha)
        }

        return listOf(addTensorGrad, tensor1Grad, tensor2Grad)
    }
}

class Dot(
    ops: TensorOperations<AbstractRawTensor<Any>>
): BiTensorFunction(ops) {

    override fun invoke(a: Tensor, b: Tensor): BiTensorFunction {
        cachedShape = a.shape.dotShape(b.shape)
        return super.invoke(a, b)
    }

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        saveForBackward(a, b)
        output = a.dot(b)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (vector1, vector2) = valuesSaved
        return listOf(
            if (inputs.first is AGTensor && inputs.first.requiresGrad)
                vector2 * gradient.broadcastTo(bShape)
            else null,
            if (inputs.second is AGTensor && inputs.second.requiresGrad)
                vector1 * gradient.broadcastTo(aShape)
            else null
        )
    }
}
