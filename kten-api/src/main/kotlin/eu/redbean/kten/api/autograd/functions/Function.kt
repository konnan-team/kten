package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
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

    private val registeredAsInputIn = mutableListOf<Function>()

    private var gradCache: AbstractRawTensor<Any>? = null

    private var backwardRan: Boolean = false

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

        registerInputUsages(getInputsAsList())

        ops.garbageCollector().use {
            if (hasValue().not())
                internalForward()

            backwardWithGrad(ops.createRaw(listOf(1)) { 1f })
            releaseUnusedInGraph()
        }

    }

    override fun backward(gradients: Tensor) {
        if (shape != gradients.shape)
            throw IllegalArgumentException(
                "Backward with gradients require same shape for gradients as the calculation result shape. " +
                        "Result shape: $shape provided gradients shape: ${gradients.shape}"
            )

        registerInputUsages(getInputsAsList())

        ops.garbageCollector().use {
            if (hasValue().not())
                internalForward()

            if (gradients is Function && gradients.hasValue().not())
                gradients.forward()

            backwardWithGrad(gradients.getRawValue())
            releaseUnusedInGraph()
        }
    }

    private fun registerInputUsages(inputs: List<Tensor>) {
        inputs.filterIsInstance<Function>()
            .forEach {
                it.registeredAsInputIn.add(this)
                it.registerInputUsages(it.getInputsAsList())
            }
    }

    private fun clearInputUsages() {
        getInputsAsList().filterIsInstance<Function>().forEach {
            it.clearInputUsages()
            it.registeredAsInputIn.clear()
        }
    }

    private fun allOutputCalculationsBackPropagated() = registeredAsInputIn.all(Function::backwardRan)

    override fun backwardWithGrad(gradient: AbstractRawTensor<Any>) {
        if (registeredAsInputIn.size > 1) { //TODO check again because it is still buggy somehow (model layer grad aggr. gives different result and faster)
            if (gradCache == null) {
                gradCache = gradient //.copy()
            } else {
                gradCache!!.plusAssign(gradient)
            }
            //ops.release(gradient)

            if (allOutputCalculationsBackPropagated().not())
                return
        }

        val gradients: List<AbstractRawTensor<Any>?>
        if (gradCache != null) {
            gradients = doBackward(gradCache!!)
        } else {
            gradients = doBackward(gradient)
            ops.release(gradient)
        }

        backwardRan = true

        gradients.forEach {
            if (it != null)
                nanCheck(it)
        }

        backwardToInputs(gradients)

        if (gradCache != null)
            ops.release(gradCache!!)
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

    internal open fun internalForward() {
        gradCache = null
        backwardRan = false
    }

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
        throw IllegalStateException(
            "Gradients are only available on variables (graph edges). " +
                    "Calculation results can be converted to variables with gradients by calling the retainGrad() method " +
                    "as part of the calculation graph, and after the backward() call using the asVariable(requiresGrad = true) method"
        )
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
            releaseUnusedInGraph()
    }

    override fun incrementRef() {
        if (hasValue())
            ops.incrementRef(output!!)
    }

}