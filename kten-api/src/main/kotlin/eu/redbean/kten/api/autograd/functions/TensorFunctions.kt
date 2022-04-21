package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.autograd.utils.*
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

@Suppress("UNCHECKED_CAST")
private operator fun AbstractRawTensor<Any>.get(index: Any): AbstractRawTensor<Any> {
    if (index is IntArray)
        return this[index]
    else
        return this[index as Array<IntRange>]
}

@Suppress("UNCHECKED_CAST")
private operator fun AbstractRawTensor<Any>.set(index: Any, value: AbstractRawTensor<Any>) {
    if (index is IntArray)
        this[index] = value
    else
        this[index as Array<IntRange>] = value
}

@Suppress("UNCHECKED_CAST")
private operator fun AbstractRawTensor<Any>.set(index: Any, value: Float) {
    if (index is IntArray)
        this[index] = value
    else
        this[index as Array<IntRange>] = value
}

abstract class AbstractIndexingFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    protected lateinit var index: Any
    protected lateinit var inputShape: List<Int>

    operator fun invoke(tensor: Tensor, index: IntArray): AbstractIndexingFunction {
        val (normalizedIndex, indexedShape) = tensor.shape.normalizedIndexedShape(index)
        if (modifiesShape)
            this.cachedShape = indexedShape
        invoke(tensor)
        this.index = normalizedIndex
        return this
    }

    operator fun invoke(tensor: Tensor, index: Array<out IntRange>): AbstractIndexingFunction {
        val (normalizedIndex, indexedShape) = tensor.shape.normalizedIndexedShape(index)
        this.cachedShape = indexedShape
        invoke(tensor)
        this.index = normalizedIndex
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        inputShape = input.shape
    }
}

class Index(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : AbstractIndexingFunction(ops) {

    init {
        modifiesShape = true
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        super.doForward(input)
        output = input[index]
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val gradInput = ops.createRaw(inputShape) { 0f }
        gradInput[index] = gradient
        return listOf(gradInput)
    }
}

class IndexSetTensor(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : BiTensorFunction(ops) {

    private lateinit var index: Any
    private lateinit var valueShape: List<Int>

    private fun checkValueShapeMatches(indexedShape: List<Int>, indexList: List<*>, tensorShape: List<Int>) {
        if (valueShape != indexedShape)
            throw IllegalArgumentException("Cannot set value with shape: ${valueShape} at index: ${indexList} on tensor with shape: ${tensorShape}")
    }

    operator fun invoke(tensor: Tensor, index: IntArray, value: Tensor): IndexSetTensor {
        val (normalizedIndex, indexedShape) = tensor.shape.normalizedIndexedShape(index)
        valueShape = value.shape
        checkValueShapeMatches(indexedShape, index.toList(), tensor.shape)
        cachedShape = tensor.shape
        invoke(tensor, value)
        this.index = normalizedIndex
        return this
    }

    operator fun invoke(tensor: Tensor, index: Array<out IntRange>, value: Tensor): IndexSetTensor {
        val (normalizedIndex, indexedShape) = tensor.shape.normalizedIndexedShape(index)
        valueShape = value.shape
        checkValueShapeMatches(indexedShape, index.toList(), tensor.shape)
        cachedShape = tensor.shape
        invoke(tensor, value)
        this.index = normalizedIndex
        return this
    }

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        ops.incrementRef(a) //To ensure not to be freed after forward
        a[index] = b
        output = a
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val gradInput = gradient.copy()
        gradInput[index] = 0.0f
        val gradValue = gradient[index].view(valueShape)
        return listOf(gradInput, gradValue)
    }

}

class IndexSetConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : AbstractIndexingFunction(ops) {

    private var value: Float = 0.0f

    private fun checkFullIndexing() {
        val indexList = if (index is IntArray) (index as IntArray).toList() else (index as Array<*>).toList()
        if (indexList.size != tensor.shape.size)
            throw IllegalArgumentException(
                "Cannot set value at index: ${indexList} on tensor with shape: ${tensor.shape} " +
                        "(All tensor dimensions must be indexed to set a constant value)"
            )
    }

    operator fun invoke(tensor: Tensor, index: IntArray, value: Float): IndexSetConstant {
        invoke(tensor, index)
        checkFullIndexing()
        this.value = value
        return this
    }

    operator fun invoke(tensor: Tensor, index: Array<out IntRange>, value: Float): IndexSetConstant {
        invoke(tensor, index)
        checkFullIndexing()
        this.value = value
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        super.doForward(input)
        ops.incrementRef(input)
        input[index] = value
        output = input
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val gradInput = gradient.copy()
        gradInput[index] = 0.0f
        return listOf(gradInput)
    }
}

class Transpose(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    private var axis1 = 0
    private var axis2 = 1

    init {
        modifiesShape = true
    }

    operator fun invoke(tensor: Tensor, axis1: Int, axis2: Int): Transpose {
        cachedShape = tensor.shape.transposeNormalizeAxes(axis1, axis2).first
        invoke(tensor)
        this.axis1 = axis1
        this.axis2 = axis2
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = input.transpose(axis1, axis2)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(gradient.transpose(axis1, axis2))
    }
}

class Reshape( // aka. View
    ops: TensorOperations<AbstractRawTensor<Any>>,
    private val viewMode: Boolean = false
) : UnaryTensorFunction(ops) {

    private lateinit var newShape: List<Int>
    private lateinit var oldShape: List<Int>

    init {
        modifiesShape = true
    }

    operator fun invoke(tensor: Tensor, newShape: List<Int>): Reshape {
        cachedShape = tensor.shape.reshape(newShape)
        invoke(tensor)
        this.newShape = newShape
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        oldShape = input.shape
        if (viewMode) {
            output = input.view(newShape)
        } else {
            output = input.reshape(newShape)
        }
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(if (viewMode) gradient.view(oldShape) else gradient.reshape(oldShape))
    }
}

class Expand( // explicit broadcast
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    private lateinit var newShape: List<Int>
    private var unsqueezedDims = 0
    private lateinit var expandedDims: List<Int>

    init {
        modifiesShape = true
    }

    operator fun invoke(tensor: Tensor, newShape: List<Int>): Expand {
        cachedShape = tensor.shape.inferExplicitBroadcastShape(newShape)
        invoke(tensor)
        this.newShape = newShape
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = input.broadcastTo(newShape)
        unsqueezedDims = newShape.size - input.dimensions
        expandedDims = newShape.drop(unsqueezedDims)
            .zip(input.shape)
            .mapIndexed { index, (expanded, original) -> index to (expanded != original) }
            .filter { it.second }
            .map { it.first }
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        var gradInput = gradient

        var gradToRelease: AbstractRawTensor<Any>
        for (i in 0 until unsqueezedDims) {
            gradToRelease = gradInput
            gradInput = gradInput.sum(0)
            if (i > 0) {
                ops.release(gradToRelease)
            }
        }

        for ((i, it) in expandedDims.withIndex()) {
            gradToRelease = gradInput
            gradInput = gradInput.sum(it, true)
            if (i > 0) {
                ops.release(gradToRelease)
            }
        }

        return listOf(gradInput)
    }
}

class Permute(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    private lateinit var axes: List<Int>
    private lateinit var reverseShapeIndices: List<Int>

    init {
        modifiesShape = true
    }

    operator fun invoke(tensor: Tensor, axes: List<Int>): Permute {
        cachedShape = tensor.shape.permute(axes)
        invoke(tensor)
        this.axes = axes
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        reverseShapeIndices = List(axes.size, axes::indexOf)
        output = input.permute(axes)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(gradient.permute(reverseShapeIndices))
    }
}

//TODO IndexAdd / IndexCopy / IndexFill (scatter like functions, I'm not sure if we need them just yet)

class IndexSelect(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    private var axis: Int = 0
    private lateinit var index: Tensor
    private lateinit var originalTensorShape: List<Int>

    operator fun invoke(tensor: Tensor, index: Tensor, axis: Int): IndexSelect {
        if (index is AGTensor && index.requiresGrad)
            throw IllegalArgumentException("Index select cannot differentiate the index. " +
                    "The index tensor passed to index select should never require gradients.")
        val (normAxis, outputShape) = tensor.shape.indexSelectNormAxisShape(axis, index.shape)
        this.modifiesShape = true
        invoke(tensor)
        this.cachedShape = outputShape
        this.originalTensorShape = tensor.shape
        this.axis = normAxis
        this.index = index
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = input.indexSelect(axis, index.getRawValue())
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val gradInput = ops.createRaw(originalTensorShape) { 0f }
        gradInput.indexAdd(axis, index.getRawValue(), gradient)
        return listOf(gradInput)
    }

}

class Concat(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : AnyTensorFunction(ops) {

    private var axis = 0
    private lateinit var inputSizes: List<Int>

    operator fun invoke(axis: Int, inputTensors: List<Tensor>): Concat {
        this.cachedShape = concatShapes(inputTensors.map { it.shape }, axis)
        this.inputTensors = inputTensors
        this.axis = cachedShape!!.normalizeAxis(axis)
        return this
    }

    override fun doForward(inputs: List<AbstractRawTensor<Any>>) {
        inputSizes = inputs.map { it.shape[axis] }
        output = ops.concat(axis, inputs)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return inputSizes.asSequence()
            .zip(inputSizes.accumulate(0, Int::plus))
            .map { (size, end) -> gradient.narrow(axis, end - size, size) }
            .toList()
    }
}

class Copy(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = input.copy()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(gradient)
    }
}

class Squeeze(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    private var axis = 0
    private lateinit var inputShape: List<Int>

    init {
        modifiesShape = true
    }

    operator fun invoke(tensor: Tensor, axis: Int): Squeeze {
        cachedShape = tensor.shape.squeeze(axis)
        invoke(tensor)
        this.axis = axis
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        this.inputShape = input.shape
        output = input.squeeze(axis)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(gradient.reshape(inputShape))
    }
}

class Unsqueeze(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    private var axis = 0

    init {
        modifiesShape = true
    }

    operator fun invoke(tensor: Tensor, axis: Int): Unsqueeze {
        cachedShape = tensor.shape.unsqueeze(axis)
        invoke(tensor)
        this.axis = axis
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = input.unsqueeze(axis)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(gradient.squeeze(axis))
    }
}

class Gather(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    private var axis = 0
    private lateinit var index: Tensor
    private lateinit var inputShape: List<Int>

    init {
        modifiesShape = true
    }

    operator fun invoke(tensor: Tensor, axis: Int, index: Tensor): Gather {
        if (index is AGTensor && index.requiresGrad)
            throw IllegalArgumentException("Gather cannot differentiate the index. The index tensor passed to gather should never require gradients.")
        //TODO calc shape
        invoke(tensor)
        this.axis = axis
        this.index = index
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        inputShape = input.shape
        val indexRaw = index.getRawValue()
        saveForBackward(indexRaw)
        output = input.gather(axis, indexRaw)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val gradInput = ops.createRaw(inputShape) { 0f }
        val (index) = valuesSaved
        gradInput.inplaceScatterAdd(axis, index, gradient)
        return listOf(gradInput)
    }
}

class Scatter(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : BiTensorFunction(ops) {

    private var axis = 0
    private lateinit var index: Tensor

    operator fun invoke(tensor: Tensor, axis: Int, index: Tensor, source: Tensor): Scatter {
        if (index is AGTensor && index.requiresGrad)
            throw IllegalArgumentException("Scatter cannot differentiate the index. The index tensor passed to scatter should never require gradients.")

        if (index.shape != source.shape) //TODO logger
            println("WARNING: backward pass is only implemented for index.shape == source.shape case, and will throw exception if called.")

        invoke(tensor, source)
        this.cachedShape = tensor.shape
        this.axis = axis
        this.index = index
        return this
    }

    @Suppress("PARAMETER_NAME_CHANGED_ON_OVERRIDE")
    override fun doForward(tensor: AbstractRawTensor<Any>, source: AbstractRawTensor<Any>) {
        val indexRaw = index.getRawValue()
        saveForBackward(indexRaw)
        output = tensor.scatter(axis, indexRaw, source)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (index) = valuesSaved

        if (index.shape != bShape)
            throw IllegalArgumentException("Index shape: ${index.shape} is not compatible with source shape: ${bShape}")

        val gradInput =
            if (inputs.first is AGTensor && inputs.first.requiresGrad)
                gradient.scatter(axis, index, 0.0f)
            else null

        val gradSource =
            if (inputs.second is AGTensor && inputs.second.requiresGrad)
                gradient.gather(axis, index)
            else null

        return listOf(gradInput, gradSource)
    }
}

class RetainGrad(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    internal var savedGradient: AbstractRawTensor<Any>? = null

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = input
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        savedGradient = gradient.copy()
        ops.incrementRef(savedGradient!!)
        return listOf(gradient)
    }

    override fun release() {
        super.release()
        if (savedGradient != null)
            ops.release(savedGradient!!)
    }

    override fun incrementRef() {
        super.incrementRef()
        if (savedGradient != null)
            ops.incrementRef(savedGradient!!)
    }

}

// TODO there are some other functions that maybe useful, but right now I don't feel the need for them
