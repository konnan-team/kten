package eu.redbean.kten.api.autograd.functions.nn

import eu.redbean.kten.api.autograd.functions.UnaryTensorFunction
import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

class SoftmaxCrossEntropy(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    private lateinit var targets: Tensor

    operator fun invoke(logits: Tensor, targets: Tensor): SoftmaxCrossEntropy {
        modifiesShape = true
        super.invoke(if (logits is AGTensor) logits.inplaceSafe() else logits)
        cachedShape = listOf(1)

        if (targets.requiresGrad) {
            throw IllegalArgumentException("SoftmaxCrossEntropy only accept targets that don't require gradients")
        }
        if (logits.dimensions != 2) {
            throw IllegalArgumentException("SoftmaxCrossEntropy requires 2D logits, but got tensor with shape ${logits.shape}")
        }
        if (targets.dimensions != 1) {
            throw IllegalArgumentException("SoftmaxCrossEntropy requires 1D targets, but got tensor with shape ${targets.shape}")
        }
        if (logits.shape[0] != targets.shape[0]) {
            throw IllegalArgumentException("Logits must have the same size at the first dimension as targets, " +
                    "but got tensors with shape: logits: ${logits.shape}, targets: ${targets.shape}")
        }

        this.targets = targets.unsqueeze(-1)

        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        val maxVal = input.max(-1, keepDimensions = true)
        val inputMinusMax = input - maxVal
        val expVal = inputMinusMax.exp()
        val expValSum = expVal.sum(-1, keepDimensions = true)

        val softmaxRes = expVal / expValSum
        saveForBackward(softmaxRes)

        val softmaxSum = softmaxRes.sum(-1, keepDimensions = true)
        val softmaxNorm = softmaxRes / softmaxSum
        val softmaxNormClip = softmaxNorm.clamp(PlatformProvider.epsilon, 1f - PlatformProvider.epsilon)
        val logSoftmax = softmaxNormClip.log()
        val logSoftmaxAtTargets = logSoftmax.gather(-1, targets.getRawValue())
        val negativeLogLikelihood = -logSoftmaxAtTargets

        output = negativeLogLikelihood.mean()

        ops.release(maxVal, inputMinusMax, expVal, expValSum, softmaxSum, softmaxNorm, softmaxNormClip, logSoftmax, logSoftmaxAtTargets, negativeLogLikelihood)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val gradIn = ops.zerosLike(valuesSaved[0])
        gradIn.inplaceScatter(-1, targets.getRawValue(), -1f)
        gradIn += valuesSaved[0]
        gradIn /= targets.shape[0].toFloat()
        return listOf(gradIn)
    }

}
