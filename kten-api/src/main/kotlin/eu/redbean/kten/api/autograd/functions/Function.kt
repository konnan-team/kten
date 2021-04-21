package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class Function(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : AGTensor(ops) {

    protected var keepOutput = false

    protected val valuesSaved = mutableListOf<AbstractRawTensor<Any>>()

    protected var output: AbstractRawTensor<Any>? = null
        set(value) {
            if (value != null)
                ops.incrementRef(value)
            field = value
        }

    private var freeOutput = false

    protected var cachedShape: List<Int>? = null

    private val nanCheck = false

    override val shape: List<Int>
        get() {
            if (cachedShape == null) {
                if (output == null)
                    internalForward()
                cachedShape = output!!.shape
            }
            return cachedShape!!
        }

    internal fun hasValue(): Boolean = output != null

    override fun forward() {
        keepOutput = true
        ops.garbageCollector().use {
            internalForward()
        }
    }

    override fun backward() {
        if (shape.size != 1 || shape[0] != 1)
            throw IllegalStateException("Backward is only allowed on singleton calculation results (tensors with shape [1])")

        ops.garbageCollector().use {
            if (hasValue().not())
                internalForward()

            backwardWithGrad(ops.createRaw(listOf(1)) { 1f })
            releaseUnusedInGraph()
        }
    }

    override fun backward(gradients: Tensor) {
        if (shape != gradients.shape)
            throw IllegalArgumentException("Backward with gradients require same shape for gradients as the calculation result shape. " +
                    "Result shape: $shape provided gradients shape: ${gradients.shape}")

        ops.garbageCollector().use {
            if (hasValue().not())
                internalForward()

            if (gradients is Function && gradients.hasValue().not())
                gradients.forward()

            backwardWithGrad(gradients.getRawValue())
            releaseUnusedInGraph()
        }
    }

    override fun backwardWithGrad(gradient: AbstractRawTensor<Any>) {
        val gradients = doBackward(gradient)

        gradients.forEach {
            if (it != null)
                nanCheck(it)
        }

        backwardToInputs(gradients)
    }

    internal open fun releaseUnusedInGraph() {
        if (freeOutput && output != null) {
            ops.release(output!!)
            output = null
            freeOutput = false
        }
        valuesSaved.forEach(ops::release)
        valuesSaved.clear()
        getInputsAsList().filter { it is Function }.map { it as Function }.forEach { it.releaseUnusedInGraph() }
    }

    abstract fun getInputsAsList(): List<Tensor>

    internal fun backwardToInputs(gradients: List<AbstractRawTensor<Any>?>) {
        getInputsAsList().zip(gradients).forEach {
            if (it.first is AGTensor && it.first.requiresGrad)
                (it.first as AGTensor).backwardWithGrad(it.second!!)
        }
    }

    internal abstract fun internalForward()

    internal fun saveForBackward(vararg rawValues: AbstractRawTensor<Any>) {
        if (hasValue().not() || valuesSaved.isEmpty())
            valuesSaved += rawValues.onEach { ops.incrementRef(it) }
    }

    override fun getRawValue(): AbstractRawTensor<Any> = output!!

    protected abstract fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?>

    internal fun mayFreeOutput() {
        if (!keepOutput) {
            freeOutput = true
        }
    }

    override fun grad(): Tensor {
        throw IllegalStateException("Gradients are only available on variables (graph edges). " +
                "Calculation results can be converted to variables with gradients by calling the retainGrad() method " +
                "as part of the calculation graph, and after the backward() call using the asVariable(requiresGrad = true) method")
    }

    protected fun nanCheck(tensor: AbstractRawTensor<Any>) {
        if (nanCheck && tensor.containsNan()) //TODO configure from platform with severity
            throw IllegalStateException("NaN found in function: ${this::class.simpleName}")
    }

    override fun toPlatform(platform: String): Tensor {
        throw UnsupportedOperationException("Functions cannot be transformed to another platform directly.")
    }

    override fun release() {
        if (keepOutput && hasValue())
            ops.release(output!!)
    }

    override fun incrementRef() {
        if (hasValue())
            ops.incrementRef(output!!)
    }

}