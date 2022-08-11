package eu.redbean.kten.api.tensor

import eu.redbean.kten.api.tensor.Tensor.Companion.tensorOf
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.platform.PlatformProvider.epsilon
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.serialization.TensorSerializer
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

internal enum class BasicOperators {
    PLUS, MINUS, TIMES, DIV, POW
}

/**
 * Tensor is a multidimensional array, with it's scalar components stored in a regular one dimensional array.
 * The scalar components of a Tensor are Float values.
 *
 * The `Tensor` class is an abstract class, which is the main API for all tensor operations.
 * Tensors can be constructed by multiple methods from the Tensor's companion object, for example
 * creating a 2x2 matrix with the invoke operator:
 *
 * ```
 *
 *  val tensor = Tensor(2, 2)
 *  println(tensor)
 *
 * ```
 * Output:
 * ```
 *  [[0.0, 0.0],
 *  [0.0, 0.0]]
 *
 * ```
 *
 * Tensors also can be created by providing it's actual values with the [tensorOf] method:
 * ```
 *
 *  val tensor = tensorOf(0.1, 0.2, 0.3)
 *  println(tensor)
 *
 * ```
 * Output:
 * ```
 *  [0.1, 0.2, 0.3]
 *
 * ```
 *
 * `Tensor` instances can be operated on with the methods provided in this class, and with the supported operator overloads,
 * for example adding two tensors together:
 *
 * ```
 *
 *  val t1 = tensorOf(1, 2, 3).expand(2, 3)
 *  println("t1:")
 *  println(t1)
 *  val t2 = tensorOf(0.1, 0.2, 0.3)
 *  println("t1 + t2:")
 *  println(t1 + t2)
 *
 * ```
 * Output:
 * ```
 *  t1:
 *  [[1.0, 2.0, 3.0],
 *  [1.0, 2.0, 3.0]]
 *  t1 + t2:
 *  [[1.1, 2.2, 3.3],
 *  [1.1, 2.2, 3.3]]
 *
 * ```
 *
 * Tensors also can be created as automatically differentiated variables, with the `requiresGrad` argument of the constructing function,
 * or converting an existing tensor with the [asVariable] method.
 * Automatic differentiation means Kten will record all operations on these variables, and calculates the gradients of them by calling
 * the [backward] method. (Backward can be called without parameters on singleton tensors, or with a gradient tensor,
 * for more details see: [backward].)
 * After the backward call the gradients of the variables can be accessed with the [grad] function.
 *
 * For example, calculating the gradient of Σ x² with respect to x:
 * ```
 *
 *  val x = tensorOf(0.1, 1.2, 2.3, requiresGrad = true)
 *  val result = sum(x pow 2)
 *  println("result: $result")
 *  result.backward()
 *  println("gradient: ${x.grad()}")
 *
 * ```
 * Output:
 * ```
 *  result: [6.74]
 *  gradient: [0.2, 2.4, 4.6]
 *
 * ```
 *
 * To make automatic differentiation possible all tensor operations give tensors as results, even the indexing operations
 * where the indexing refers to a single scalar value will result a singleton tensor (tensor with a single value).
 * The actual value of a singleton tensor is accessible with the [item] function.
 *
 * For example:
 * ```
 *
 *  val tensor = tensorOf(1, 2, 3)
 *  val valueAt1 = tensor[1]
 *  println("Value: $valueAt1 type: ${valueAt1::class.simpleName}")
 *
 *  val item = valueAt1.item()
 *  println("Value: $item type: ${item::class.simpleName}")
 *
 * ```
 * Output:
 * ```
 *  Value: [2.0] type: NoGradVariable
 *  Value: 2.0 type: Float
 * ```
 */
abstract class Tensor(
    protected val ops: TensorOperations<AbstractRawTensor<Any>>
) {

    /**
     * Describes the number of elements in each dimension.
     *
     * For example a tensor representing a matrix with shape `[2, 3]` has 2 rows and 3 columns.
     */
    abstract val shape: List<Int>

    /**
     * The number of dimensions the tensor has. (For example a matrix has 2 dimensions.)
     */
    val dimensions: Int
        get() = shape.size

    /**
     * Describes that the tensor requires gradient calculations in [backward] or not
     */
    abstract val requiresGrad: Boolean

    /**
     * The platform key of the tensor data. (For more details see: [toPlatform])
     */
    abstract val platform: String

    /**
     * Gets the scalar component of a singleton tensor (tensor with a single value)
     */
    abstract fun item(): Float

    /**
     * Gets the scalar component of the tensor at the specified index
     *
     * @param index The index of the scalar component
     * @throws IllegalArgumentException If the specified index doesn't indexes a single value
     */
    abstract fun getValue(index: List<Int>): Float

    /**
     * Gets the tensor at the specified index.
     *
     * Index values may be negative numbers representing the `<number of elements> - |<index value>|`
     * at the position of the index value.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(8).reshape(2, 2, 2)
     * println(tensor)
     * println()
     * println("tensor[0, -1]: ${tensor[0, -1]}")
     * ```
     * Output:
     * ```
     * [[[0.0, 1.0],
     * [2.0, 3.0]],
     * [[4.0, 5.0],
     * [6.0, 7.0]]]
     *
     * tensor[0, -1]: [2.0, 3.0]
     * ```
     */
    abstract operator fun get(vararg index: Int): Tensor

    /**
     * Gets the "slice" of the tensor specified by the index. This allows the advanced indexing options for tensors.
     *
     * The index values may have negative number as range boundaries, representing the `shape[axis] - |<range boundary>|`,
     * where `axis` is the position of the index value.
     *
     * In addition the range constant [Constants.all] can be used to specify all elements at the index value position.
     *
     * Example:
     * ```
     *  val tensor = Tensor.arrange(27).reshape(3, 3, 3)
     *  println(tensor)
     *  println()
     *  println("tensor[all, 1..-1, 1..1]:")
     *  println(tensor[all, 1..-1, 1..1])
     * ```
     * Output:
     * ```
     *  [[[0.0, 1.0, 2.0],
     *  [3.0, 4.0, 5.0],
     *  [6.0, 7.0, 8.0]],
     *  [[9.0, 10.0, 11.0],
     *  [12.0, 13.0, 14.0],
     *  [15.0, 16.0, 17.0]],
     *  [[18.0, 19.0, 20.0],
     *  [21.0, 22.0, 23.0],
     *  [24.0, 25.0, 26.0]]]
     *
     * tensor[all, 1..-1, 1..1]:
     * [[[4.0],
     * [7.0]],
     * [[13.0],
     * [16.0]],
     * [[22.0],
     * [25.0]]]
     * ```
     */
    abstract operator fun get(vararg index: IntRange): Tensor

    /**
     * Sets the tensor values from another tensor at the given index.
     *
     * Index values may be negative numbers representing the `<number of elements> - |<index value>|`
     * at the position of the index value.
     *
     * Example:
     * ```
     * val t1 = tensorOf(
     *     1, 2,
     *     3, 4,
     *
     *     5, 6,
     *     7, 8
     * ).reshape(2, 2, 2)
     * val t2 = tensorOf(30, 40)
     *
     * t1[0, -1] = t2
     *
     * println(t1)
     * ```
     * Output:
     * ```
     * [[[1.0, 2.0],
     * [30.0, 40.0]],
     * [[5.0, 6.0],
     * [7.0, 8.0]]]
     * ```
     */
    abstract operator fun set(vararg index: Int, value: Tensor)

    /**
     * Sets the tensor values at the specified index ranges from another tensor.
     *
     * The index values may have negative number as range boundaries, representing the `shape[axis] - |<range boundary>|`,
     * where `axis` is the position of the index value.
     *
     * In addition the range constant [Constants.all] can be used to specify all elements at the index value position.
     *
     * Example:
     * ```
     * val t1 = Tensor.arrange(27).reshape(3, 3, 3)
     *
     * println("t1:")
     * println(t1)
     *
     * val t2 = tensorOf(-1).expand(3, 2, 1)
     *
     * t1[all, 1..-1, 1..1] = t2
     *
     * println("t1 after set:")
     * println(t1)
     * ```
     * Output:
     * ```
     * t1:
     * [[[0.0, 1.0, 2.0],
     * [3.0, 4.0, 5.0],
     * [6.0, 7.0, 8.0]],
     * [[9.0, 10.0, 11.0],
     * [12.0, 13.0, 14.0],
     * [15.0, 16.0, 17.0]],
     * [[18.0, 19.0, 20.0],
     * [21.0, 22.0, 23.0],
     * [24.0, 25.0, 26.0]]]
     *
     * t1 after set:
     * [[[0.0, 1.0, 2.0],
     * [3.0, -1.0, 5.0],
     * [6.0, -1.0, 8.0]],
     * [[9.0, 10.0, 11.0],
     * [12.0, -1.0, 14.0],
     * [15.0, -1.0, 17.0]],
     * [[18.0, 19.0, 20.0],
     * [21.0, -1.0, 23.0],
     * [24.0, -1.0, 26.0]]]
     * ```
     */
    abstract operator fun set(vararg index: IntRange, value: Tensor)

    /**
     * Sets the values of the tensor at index to the specified float value.
     *
     * Index values may be negative numbers representing the `<number of elements> - |<index value>|`
     * at the position of the index value.
     *
     * If the index doesn't correspond to a single value the whole indexed segment of the tensor will be filled
     * with the specified value.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(9).reshape(3, 3)
     * println("tensor:")
     * println(tensor)
     *
     * tensor[1, 0] = -1f
     *
     * println("tensor after single value set:")
     * println(tensor)
     *
     * tensor[-1] = -1f
     *
     * println("tensor after multiple values set:")
     * println(tensor)
     * ```
     * Output:
     * ```
     * tensor:
     * [[0.0, 1.0, 2.0],
     * [3.0, 4.0, 5.0],
     * [6.0, 7.0, 8.0]]
     *
     * tensor after single value set:
     * [[0.0, 1.0, 2.0],
     * [-1.0, 4.0, 5.0],
     * [6.0, 7.0, 8.0]]
     *
     * tensor after multiple values set:
     * [[0.0, 1.0, 2.0],
     * [-1.0, 4.0, 5.0],
     * [-1.0, -1.0, -1.0]]
     * ```
     */
    abstract operator fun set(vararg index: Int, value: Float)

    /**
     * Sets the tensor values at the given index ranges to the specified float value.
     *
     * The index values may have negative number as range boundaries, representing the `shape[axis] - |<range boundary>|`,
     * where `axis` is the position of the index value.
     *
     * In addition the range constant [Constants.all] can be used to specify all elements at the index value position.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(9).reshape(3, 3)
     * println("tensor:")
     * println(tensor)
     *
     * tensor[all, 1..-1] = -1f
     *
     * println("tensor after set:")
     * println(tensor)
     * ```
     * Output:
     * ```
     * tensor:
     * [[0.0, 1.0, 2.0],
     * [3.0, 4.0, 5.0],
     * [6.0, 7.0, 8.0]]
     *
     * tensor after set:
     * [[0.0, -1.0, -1.0],
     * [3.0, -1.0, -1.0],
     * [6.0, -1.0, -1.0]]
     * ```
     */
    abstract operator fun set(vararg index: IntRange, value: Float)

    /**
     * Alias for `set(vararg index: Int, value: Float)` with type conversion
     */
    operator fun set(vararg index: Int, value: Int) {
        this.set(*index, value = value.toFloat())
    }

    /**
     * Alias for `set(vararg index: Int, value: Float)` with type conversion
     */
    operator fun set(vararg index: Int, value: Double) {
        this.set(*index, value = value.toFloat())
    }

    /**
     * Elementwise `plus` operation on tensors, with implicit broadcasting.
     *
     * Broadcasting applies when the two operands' shape is different and the implicit broadcasting rule is applicable.
     *
     * **Implicit broadcasting rule:** A tensor is implicitly broadcastable to another shape, if all non-singleton dimensions
     * match in the tensor's shape and the desired shape, all missing dimensions are considered as singleton dimensions at the
     * beginning of the tensor's shape.
     *
     * Example:
     * ```
     * val t1 = Tensor.arrange(8).reshape(2, 2, 2)
     * println("t1:")
     * println(t1)
     *
     * val t2 = tensorOf(10, 20).reshape(2, 1)
     * println("t2:")
     * println(t2)
     *
     * println("t1 + t2:")
     * println(t1 + t2)
     * ```
     * Output:
     * ```
     * t1:
     * [[[0.0, 1.0],
     * [2.0, 3.0]],
     * [[4.0, 5.0],
     * [6.0, 7.0]]]
     *
     * t2:
     * [[10.0],
     * [20.0]]
     *
     * t1 + t2:
     * [[[10.0, 11.0],
     * [22.0, 23.0]],
     * [[14.0, 15.0],
     * [26.0, 27.0]]]
     * ```
     */
    abstract operator fun plus(other: Tensor): Tensor

    /**
     * Adds the constant value to all tensor values.
     */
    abstract operator fun plus(constant: Float): Tensor

    abstract operator fun Float.plus(tensor: Tensor): Tensor

    /**
     * Elementwise `minus` operation on tensors, with implicit broadcast.
     *
     * @see plus
     */
    abstract operator fun minus(other: Tensor): Tensor

    /**
     * Subtracts the constant from all tensor values.
     */
    abstract operator fun minus(constant: Float): Tensor

    abstract operator fun Float.minus(tensor: Tensor): Tensor

    /**
     * Elementwise `times` operation on tensors, with implicit broadcast.
     *
     * @see plus
     */
    abstract operator fun times(other: Tensor): Tensor

    /**
     * Multiplies all tensor values with the given constant.
     */
    abstract operator fun times(constant: Float): Tensor

    abstract operator fun Float.times(tensor: Tensor): Tensor

    /**
     * Elementwise `div` operation on tensors, with implicit broadcast.
     *
     * @see plus
     */
    abstract operator fun div(other: Tensor): Tensor

    /**
     * Divides all tensor values with the given constant.
     */
    abstract operator fun div(constant: Float): Tensor

    abstract operator fun Float.div(tensor: Tensor): Tensor

    /**
     * Raises the left hand side operand tensor to the power of the right hand side operand tensor elementwise, with implicit broadcasting.
     *
     * For details on implicit broadcasting see [plus]
     *
     * Example:
     * ```
     * val t1 = tensorOf(2).expand(2, 2, 2)
     * println("t1:")
     * println(t1)
     *
     * val t2 = tensorOf(2, 3).reshape(2, 1)
     * println("t2:")
     * println(t2)
     *
     * println("t1 pow t2:")
     * println(t1 pow t2)
     * ```
     * Output:
     * ```
     * t1:
     * [[[2.0, 2.0],
     * [2.0, 2.0]],
     * [[2.0, 2.0],
     * [2.0, 2.0]]]
     *
     * t2:
     * [[2.0],
     * [3.0]]
     *
     * t1 pow t2:
     * [[[4.0, 4.0],
     * [8.0, 8.0]],
     * [[4.0, 4.0],
     * [8.0, 8.0]]]
     * ```
     *
     * @see plus
     */
    abstract infix fun pow(other: Tensor): Tensor

    /**
     * Raises all tensor values to the power of the given constant.
     */
    abstract infix fun pow(constant: Float): Tensor

    /**
     * Alias for `pow(constant: Float)` with type conversion.
     */
    infix fun pow(constant: Int): Tensor = this.pow(constant.toFloat())

    abstract infix fun Float.pow(tensor: Tensor): Tensor

    internal fun constOpTensor(constant: Float, tensor: Tensor, operator: BasicOperators): Tensor {
        return when (operator) {
            BasicOperators.PLUS -> constant + tensor
            BasicOperators.MINUS -> constant - tensor
            BasicOperators.TIMES -> constant * tensor
            BasicOperators.DIV -> constant / tensor
            BasicOperators.POW -> constant pow tensor
        }
    }

    /**
     * Inverts the sign of all tensor values.
     */
    operator fun unaryMinus(): Tensor {
        return 0.0f - this
    }

    /**
     * Calculates the sum of the tensor values over the given axis.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(4).reshape(2, 2)
     * println("tensor:")
     * println(tensor)
     *
     * println("tensor.sum(axis = -1):")
     * println(tensor.sum(-1))
     *
     * println("tensor.sum(axis = -1, keepDimensions = true):")
     * println(tensor.sum(axis = -1, keepDimensions = true))
     * ```
     * Output:
     * ```
     * tensor:
     * [[0.0, 1.0],
     * [2.0, 3.0]]
     *
     * tensor.sum(axis = -1):
     * [1.0, 5.0]
     *
     * tensor.sum(axis = -1, keepDimensions = true):
     * [[1.0],
     * [5.0]]
     * ```
     *
     * @param axis specifies the axis along which the operation is performed. The axis may be negative value,
     * representing `<tensor dimensions> - |<axis value>|`.
     * @param keepDimensions if `true` the reduced axis kept in the result with size one.
     * @throws IllegalArgumentException if axis is out of bounds `-<tensor dimensions>` inclusive
     * and `<tensor dimensions>` exclusive
     */
    abstract fun sum(axis: Int, keepDimensions: Boolean = false): Tensor

    /**
     * Calculates the sum of the tensor values into a singleton tensor. (Tensor with shape: [1])
     *
     * Example:
     * ```
     * val tensor = tensorOf(1, 2, 3, 4).reshape(2, 2)
     * println(tensor)
     * println("sum of tensor: ${tensor.sum()}")
     * ```
     * Output:
     * ```
     * [[1.0, 2.0],
     * [3.0, 4.0]]
     *
     * sum of tensor: [10.0]
     * ```
     */
    abstract fun sum(): Tensor

    /**
     * Calculates the mean of the tensor values over the given axis.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(4).reshape(2, 2)
     * println("tensor:")
     * println(tensor)
     *
     * println("tensor.mean(axis = -1):")
     * println(tensor.mean(-1))
     *
     * println("tensor.mean(axis = -1, keepDimensions = true):")
     * println(tensor.mean(axis = -1, keepDimensions = true))
     * ```
     * Output:
     * ```
     * tensor:
     * [[0.0, 1.0],
     * [2.0, 3.0]]
     *
     * tensor.mean(axis = -1):
     * [0.5, 2.5]
     *
     * tensor.mean(axis = -1, keepDimensions = true):
     * [[0.5],
     * [2.5]]
     * ```
     *
     * @param axis specifies the axis along which the operation is performed. The axis may be negative value,
     * representing `<tensor dimensions> - |<axis value>|`.
     * @param keepDimensions if `true` the reduced axis kept in the result with size one.
     * @throws IllegalArgumentException if axis is out of bounds `-<tensor dimensions>` inclusive
     * and `<tensor dimensions>` exclusive
     */
    abstract fun mean(axis: Int, keepDimensions: Boolean = false): Tensor

    /**
     * Calculates the mean of the tensor values into a singleton tensor. (Tensor with shape: [1])
     *
     * Example:
     * ```
     * val tensor = tensorOf(1, 2, 3, 4).reshape(2, 2)
     * println(tensor)
     * println("mean of tensor: ${tensor.mean()}")
     * ```
     * Output:
     * ```
     * [[1.0, 2.0],
     * [3.0, 4.0]]
     *
     * mean of tensor: [2.5]
     * ```
     */
    abstract fun mean(): Tensor

    /**
     * Finds the maximum of the tensor values over the given axis.
     *
     * Example:
     * ```
     * val tensor = tensorOf(
     * 3, 1, 2,
     * 5, 6, 4,
     * 7, 8, 9
     * ).reshape(3, 3)
     * println("tensor:")
     * println(tensor)
     *
     * println("tensor.max(axis = -1):")
     * println(tensor.max(-1))
     *
     * println("tensor.max(axis = -1, keepDimensions = true):")
     * println(tensor.max(axis = -1, keepDimensions = true))
     * ```
     * Output:
     * ```
     * tensor:
     * [[3.0, 1.0, 2.0],
     * [5.0, 6.0, 4.0],
     * [7.0, 8.0, 9.0]]
     *
     * tensor.max(axis = -1):
     * [3.0, 6.0, 9.0]
     *
     * tensor.max(axis = -1, keepDimensions = true):
     * [[3.0],
     * [6.0],
     * [9.0]]
     * ```
     *
     * @param axis specifies the axis along which the operation is performed. The axis may be negative value,
     * representing `<tensor dimensions> - |<axis value>|`.
     * @param keepDimensions if `true` the reduced axis kept in the result with size one.
     * @throws IllegalArgumentException if axis is out of bounds `-<tensor dimensions>` inclusive
     * and `<tensor dimensions>` exclusive
     */
    abstract fun max(axis: Int, keepDimensions: Boolean = false): Tensor

    /**
     * Finds the minimum of the tensor values over the given axis.
     *
     * Example:
     * ```
     * val tensor = tensorOf(
     * 3, 1, 2,
     * 5, 6, 4,
     * 7, 8, 9
     * ).reshape(3, 3)
     * println("tensor:")
     * println(tensor)
     *
     * println("tensor.min(axis = -1):")
     * println(tensor.min(-1))
     *
     * println("tensor.min(axis = -1, keepDimensions = true):")
     * println(tensor.min(axis = -1, keepDimensions = true))
     * ```
     * Output:
     * ```
     * tensor:
     * [[3.0, 1.0, 2.0],
     * [5.0, 6.0, 4.0],
     * [7.0, 8.0, 9.0]]
     *
     * tensor.min(axis = -1):
     * [1.0, 4.0, 7.0]
     *
     * tensor.min(axis = -1, keepDimensions = true):
     * [[1.0],
     * [4.0],
     * [7.0]]
     * ```
     *
     * @param axis specifies the axis along which the operation is performed. The axis may be negative value,
     * representing `<tensor dimensions> - |<axis value>|`.
     * @param keepDimensions if `true` the reduced axis kept in the result with size one.
     * @throws IllegalArgumentException if axis is out of bounds `-<tensor dimensions>` inclusive
     * and `<tensor dimensions>` exclusive
     */
    abstract fun min(axis: Int, keepDimensions: Boolean = false): Tensor

    /**
     * Finds the indices for the maximum of the tensor values over the given axis, and returns them as a tensor.
     *
     * Example:
     * ```
     * val tensor = tensorOf(
     * 3, 1, 2,
     * 5, 6, 4,
     * 7, 8, 9
     * ).reshape(3, 3)
     * println("tensor:")
     * println(tensor)
     *
     * println("tensor.argMax(axis = -1):")
     * println(tensor.argMax(-1))
     *
     * println("tensor.argMax(axis = -1, keepDimensions = true):")
     * println(tensor.argMax(axis = -1, keepDimensions = true))
     * ```
     * Output:
     * ```
     * tensor:
     * [[3.0, 1.0, 2.0],
     * [5.0, 6.0, 4.0],
     * [7.0, 8.0, 9.0]]
     *
     * tensor.argMax(axis = -1):
     * [0.0, 1.0, 2.0]
     *
     * tensor.argMax(axis = -1, keepDimensions = true):
     * [[0.0],
     * [1.0],
     * [2.0]]
     * ```
     *
     * @param axis specifies the axis along which the operation is performed. The axis may be negative value,
     * representing `<tensor dimensions> - |<axis value>|`.
     * @param keepDimensions if `true` the reduced axis kept in the result with size one.
     * @throws IllegalArgumentException if axis is out of bounds `-<tensor dimensions>` inclusive
     * and `<tensor dimensions>` exclusive
     */
    abstract fun argMax(axis: Int = 0, keepDimensions: Boolean = false): Tensor

    /**
     * Finds the indices for the minimum of the tensor values over the given axis, and returns them as a tensor.
     *
     * Example:
     * ```
     * val tensor = tensorOf(
     * 3, 1, 2,
     * 5, 6, 4,
     * 7, 8, 9
     * ).reshape(3, 3)
     * println("tensor:")
     * println(tensor)
     *
     * println("tensor.argMin(axis = -1):")
     * println(tensor.argMin(-1))
     *
     * println("tensor.argMin(axis = -1, keepDimensions = true):")
     * println(tensor.argMin(axis = -1, keepDimensions = true))
     * ```
     * Output:
     * ```
     * tensor:
     * [[3.0, 1.0, 2.0],
     * [5.0, 6.0, 4.0],
     * [7.0, 8.0, 9.0]]
     *
     * tensor.argMin(axis = -1):
     * [1.0, 2.0, 0.0]
     *
     * tensor.argMin(axis = -1, keepDimensions = true):
     * [[1.0],
     * [2.0],
     * [0.0]]
     * ```
     *
     * @param axis specifies the axis along which the operation is performed. The axis may be negative value,
     * representing `<tensor dimensions> - |<axis value>|`.
     * @param keepDimensions if `true` the reduced axis kept in the result with size one.
     * @throws IllegalArgumentException if axis is out of bounds `-<tensor dimensions>` inclusive
     * and `<tensor dimensions>` exclusive
     */
    abstract fun argMin(axis: Int = 0, keepDimensions: Boolean = false): Tensor

    protected abstract fun exp(): Tensor

    protected abstract fun log(): Tensor

    protected abstract fun tanh(): Tensor

    protected abstract fun sigmoid(): Tensor

    protected abstract fun sinh(): Tensor

    protected abstract fun cosh(): Tensor

    protected abstract fun abs(): Tensor

    /**
     * Clamps the tensor values outside the given range, specified by the `min` and `max` parameters.
     *
     * This method specifies the min and max values by default for more convenient usage. (The default minimum specified as
     * `Float.NEGATIVE_INFINITY + epsilon` and the maximum as `Float.POSITIVE_INFINITY - epsilon`, where `epsilon` refers to
     * [PlatformProvider.epsilon])
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(9).reshape(3, 3)
     * println("tensor:")
     * println(tensor)
     *
     * println("clamped:")
     * println(tensor.clamp(min = 3.5f, max = 5.5f))
     * ```
     * Output:
     * ```
     * tensor:
     * [[0.0, 1.0, 2.0],
     * [3.0, 4.0, 5.0],
     * [6.0, 7.0, 8.0]]
     *
     * clamped:
     * [[3.5, 3.5, 3.5],
     * [3.5, 4.0, 5.0],
     * [5.5, 5.5, 5.5]]
     * ```
     *
     * @param min the minimum value allowed in the result
     * @param max the maximum value allowed in the result
     */
    abstract fun clamp(min: Float = Float.NEGATIVE_INFINITY + epsilon, max: Float = Float.POSITIVE_INFINITY - epsilon): Tensor

    protected abstract fun sqrt(): Tensor

    protected abstract fun sin(): Tensor

    protected abstract fun cos(): Tensor

    protected abstract fun tan(): Tensor

    protected abstract fun asin(): Tensor

    protected abstract fun acos(): Tensor

    protected abstract fun atan(): Tensor

    /**
     * Calculates the reciprocal of the tensor values.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(5f, step = 1f, start = 1f).reshape(2, 2)
     * println(tensor)
     * println("reciprocal:")
     * println(tensor.reciprocal())
     * ```
     * Output:
     * ```
     * [[1.0, 2.0],
     * [3.0, 4.0]]
     *
     * reciprocal:
     * [[1.0, 0.5],
     * [0.33333334, 0.25]]
     * ```
     */
    abstract fun reciprocal(): Tensor

    /**
     * Floors the values in the tensor.
     *
     * Example:
     * ```
     * val tensor = tensorOf(
     * 0.1, 0.5,
     * 1.4, 2.9
     * ).reshape(2, 2)
     * println(tensor.floor())
     * ```
     * Output:
     * ```
     * [[0.0, 0.0],
     * [1.0, 2.0]]
     * ```
     */
    abstract fun floor(): Tensor

    /**
     * Ceil the values of the tensor.
     *
     * Example:
     * ```
     * val tensor = tensorOf(
     * 0.1, 0.5,
     * 1.4, 2.9
     * ).reshape(2, 2)
     * println(tensor.ceil())
     * ```
     * Output:
     * ```
     * [[1.0, 1.0],
     * [2.0, 3.0]]
     * ```
     */
    abstract fun ceil(): Tensor

    /**
     * Rounds the values of the tensor.
     *
     * Example:
     * ```
     * val tensor = tensorOf(
     * 0.1, 0.5,
     * 1.4, 2.9
     * ).reshape(2, 2)
     * println(tensor.round())
     * ```
     * Output:
     * ```
     * [[0.0, 1.0],
     * [1.0, 3.0]]
     * ```
     */
    abstract fun round(): Tensor

    /**
     * Returns a tensor with the sign values for `this` tensor.
     *
     * The resulting tensor will have the same shape as `this` tensor, and the scalar components of the result will be
     * -1 if the scalar component of `this` is negative at the same position, 0 if it is zero and 1 if it is positive.
     *
     * Example:
     * ```
     * var tensor = arrange(9).reshape(3, 3)
     * tensor = tensor - mean(tensor)
     * println(tensor)
     * println(tensor.sign())
     * ```
     * Output:
     * ```
     * [[-4.0, -3.0, -2.0],
     * [-1.0, 0.0, 1.0],
     * [2.0, 3.0, 4.0]]
     *
     * [[-1.0, -1.0, -1.0],
     * [-1.0, 0.0, 1.0],
     * [1.0, 1.0, 1.0]]
     * ```
     */
    abstract fun sign(): Tensor

    /**
     * Truncates the fractions of the tensor values.
     *
     * Example:
     * ```
     * val tensor = tensorOf(
     * 0.1, 0.5,
     * 1.4, 2.9
     * ).reshape(2, 2)
     * println(tensor.trunc())
     * ```
     * Output:
     * ```
     * [[0.0, 0.0],
     * [1.0, 2.0]]
     * ```
     */
    abstract fun trunc(): Tensor

    protected abstract fun rsqrt(): Tensor

    /**
     * Transposes the tensor along the given two axes.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(8).reshape(2, 2, 2)
     * println(tensor)
     * println("transposed:")
     * println(tensor.transpose(1, -1))
     * ```
     * Output:
     * ```
     * [[[0.0, 1.0],
     * [2.0, 3.0]],
     * [[4.0, 5.0],
     * [6.0, 7.0]]]
     *
     * transposed:
     * [[[0.0, 2.0],
     * [1.0, 3.0]],
     * [[4.0, 6.0],
     * [5.0, 7.0]]]
     * ```
     *
     * @param axis1 first axis to transpose along, accepts negative value, which represents `<tensor dimensions> - |axis1|`.
     * @param axis2 second axis to transpose along, accepts negative value, which represents `<tensor dimensions> - |axis1|`.
     * @throws IllegalArgumentException if either axis1 or axis2 is out of bounds `-<tensor dimensions>` inclusive
     * and `<tensor dimensions>` exclusive, or if the axes are pointing to the same tensor axis.
     */
    abstract fun transpose(axis1: Int, axis2: Int): Tensor

    /**
     * Creates a new tensor with the given shape and a copy of the tensor values of this tensor.
     *
     * The new shape for the result must describe the same number of elements as the original shape
     * (`(∏ nesShape[i] for i in 0..newShape.size) == (∏ shape[j] for j in 0..shape.size)`).
     *
     * The new shape may contain a single wildcard element represented as `-1`, which means the shape size at the position
     * of the wildcard value will be inferred from the tensor size and the other shape values. (If only the wildcard value `-1`
     * is passed to the reshape, the operation will essentially flatten the tensor.)
     *
     * Example:
     * ```
     * var tensor = Tensor.arrange(8)
     * println(tensor)
     * println("reshape to 2x2x2:")
     * tensor = tensor.reshape(2, 2, 2)
     * println("$tensor shape: ${tensor.shape}")
     * println("reshape with wildcard:")
     * tensor = tensor.reshape(2, -1)
     * println("$tensor shape: ${tensor.shape}")
     * ```
     * Output:
     * ```
     * [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
     * reshape to 2x2x2:
     *
     * [[[0.0, 1.0],
     * [2.0, 3.0]],
     * [[4.0, 5.0],
     * [6.0, 7.0]]] shape: [2, 2, 2]
     * reshape with wildcard:
     *
     * [[0.0, 1.0, 2.0, 3.0],
     * [4.0, 5.0, 6.0, 7.0]] shape: [2, 4]
     * ```
     *
     * @param newShape the required shape for the tensor.
     * @throws IllegalArgumentException if the given new shape doesn't describes the same number of elements, contains more than
     * one wildcard value or the inferred shape is not valid for the tensor (For example a tensor with shape `[2, 4]` cannot be
     * reshaped with `(3, -1, 2)` parameters because `2*4` is not divisible with `2*3`.).
     */
    fun reshape(vararg newShape: Int): Tensor = reshape(newShape.asList())

    /**
     * Same as `reshape(vararg newShape: Int)` with `newShape` as list.
     *
     * @see reshape
     */
    abstract fun reshape(newShape: List<Int>): Tensor

    /**
     * Expands the tensor to the given new shape (by repeating the values in the expanded dimensions),
     * if the new shape complies with the broadcasting rule.
     *
     * **Broadcasting rule:** A tensor is broadcastable to another shape, if all non-singleton dimensions match in the tensor's shape
     * with the desired shape, all missing dimensions are considered as singleton dimensions at the beginning of the tensor's shape,
     * and `newShape.size >= <tensor dimensions>`.
     *
     * Example:
     * ```
     * var tensor = Tensor.arrange(8).reshape(2, 1, 4)
     * println(tensor)
     * println("expanded:")
     * println(tensor.expand(2, 2, 3, 4))
     * ```
     * Output:
     * ```
     * [[[0.0, 1.0, 2.0, 3.0]],
     * [[4.0, 5.0, 6.0, 7.0]]]
     *
     * expanded:
     * [[[[0.0, 1.0, 2.0, 3.0],
     * [0.0, 1.0, 2.0, 3.0],
     * [0.0, 1.0, 2.0, 3.0]],
     * [[4.0, 5.0, 6.0, 7.0],
     * [4.0, 5.0, 6.0, 7.0],
     * [4.0, 5.0, 6.0, 7.0]]],
     * [[[0.0, 1.0, 2.0, 3.0],
     * [0.0, 1.0, 2.0, 3.0],
     * [0.0, 1.0, 2.0, 3.0]],
     * [[4.0, 5.0, 6.0, 7.0],
     * [4.0, 5.0, 6.0, 7.0],
     * [4.0, 5.0, 6.0, 7.0]]]]
     * ```
     *
     * @param newShape the desired shape for the tensor.
     * @throws IllegalArgumentException if the tensor is not broadcastable to the given `newShaoe`.
     */
    abstract fun expand(newShape: List<Int>): Tensor

    /**
     * Same as `expand(newShape: List<Int>)` with `newShape` as `vararg`.
     *
     * @see expand
     */
    fun expand(vararg newShape: Int): Tensor = expand(newShape.toList())

    /**
     * Permutes the tensor axes to the given arrangement.
     *
     * **Note:** All tensor axes must be specified in the `axes` parameter and negative values are not allowed.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(8).reshape(2, 2, 2)
     * println(tensor)
     * println("permuted:")
     * println(tensor.permute(2, 1, 0))
     * ```
     * Output:
     * ```
     * [[[0.0, 1.0],
     * [2.0, 3.0]],
     * [[4.0, 5.0],
     * [6.0, 7.0]]]
     *
     * permuted:
     * [[[0.0, 4.0],
     * [2.0, 6.0]],
     * [[1.0, 5.0],
     * [3.0, 7.0]]]
     * ```
     *
     * @param axes The desired tensor axes arrangement.
     * @throws IllegalArgumentException if not all axes are specified.
     */
    abstract fun permute(axes: List<Int>): Tensor

    /**
     * Same as `permute(axes: List<Int>)` with axes as vararg.
     *
     * @see permute
     */
    fun permute(vararg axes: Int): Tensor = permute(axes.toList())

    protected abstract fun concat(axis: Int, tensors: List<Tensor>): Tensor

    /**
     * Creates an identical copy of the tensor.
     */
    abstract fun copy(): Tensor

    /**
     * Removes the specified axis from the tensor, the axis must refer to a singleton dimension (`shape[axis] == 1`).
     *
     * Example:
     * ```
     * var tensor = Tensor.arrange(4).reshape(2, 1, 2)
     * tensor = tensor.squeeze(axis = 1)
     * println("$tensor shape: ${tensor.shape}")
     * ```
     * Output:
     * ```
     * [[0.0, 1.0],
     * [2.0, 3.0]] shape: [2, 2]
     * ```
     *
     * @param axis specifies the axis to remove. The axis may be negative value, representing `<tensor dimensions> - |<axis value>|`.
     * @throws IllegalArgumentException if the specified axis is not in the tensor, or if the the size at the specified axis isn't one.
     */
    abstract fun squeeze(axis: Int): Tensor

    /**
     * Adds a singleton dimension (`resultTensor.shape[axis] == 1`) to the tensor at the specified axis.
     *
     * Example:
     * ```
     * var tensor = Tensor.arrange(4).reshape(2, 2)
     * tensor = tensor.unsqueeze(1)
     * println("$tensor shape: ${tensor.shape}")
     * ```
     * Output:
     * ```
     * [[[0.0, 1.0]],
     * [[2.0, 3.0]]] shape: [2, 1, 2]
     * ```
     * @param axis specifies the axis to add. The axis may be negative value, representing `<tensor dimensions> - |<axis value>| + 1`.
     * @throws IllegalArgumentException if the specified axis is out of bounds for `(<tensor dimensions> + 1) * -1` inclusive and
     * `<tensor dimensions>` inclusive.
     */
    abstract fun unsqueeze(axis: Int): Tensor

    /**
     * Gathers values from the tensor along the specified [axis], indexed by the [index] values.
     * The index tensor must have the same shape as the tensor, for all axis, except the specified axis. The result will have the same
     * shape as the index.
     *
     * Values from the [index] tensor are converted to integer with simple casting, and invalid index value handling (outside the bounds
     * of the 0 to `shape[axis]` range) is backend implementation specific.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(27).reshape(3, 3, 3)
     * val index = tensorOf(
     * 0, 1, 2,
     * 2, 1, 0,
     *
     * 2, 0, 1,
     * 1, 0, 2,
     *
     * 0, 2, 1,
     * 1, 2, 0
     * ).reshape(3, 2, 3)
     *
     * var res1 = Tensor.zerosLike(index)
     *
     * for (i in 0 until index.shape[0])
     *     for (j in 0 until index.shape[1])
     *         for (k in 0 until index.shape[2])
     *             res1[i, j, k] = tensor[i, index[i, j, k].item().toInt(), k]
     *
     * val res2 = tensor.gather(axis = 1, index = index)
     *
     * println("res1:\n$res1")
     * println("res2:\n$res2")
     * ```
     * Output:
     * ```
     * res1:
     * [[[0.0, 4.0, 8.0],
     * [6.0, 4.0, 2.0]],
     * [[15.0, 10.0, 14.0],
     * [12.0, 10.0, 17.0]],
     * [[18.0, 25.0, 23.0],
     * [21.0, 25.0, 20.0]]]
     *
     * res2:
     * [[[0.0, 4.0, 8.0],
     * [6.0, 4.0, 2.0]],
     * [[15.0, 10.0, 14.0],
     * [12.0, 10.0, 17.0]],
     * [[18.0, 25.0, 23.0],
     * [21.0, 25.0, 20.0]]]
     * ```
     * @param axis the tensor axis from, the values are gathered. The axis may be negative value, representing `<tensor dimensions> - |<axis value>|`.
     * @param index the tensor containing the index values for the specified axis. Index values should be in the range of `0` inclusive
     * and `tensor.shape[axis]` exclusive.
     * @throws IllegalArgumentException if the specified axis is not in the tensor, the index tensor has invalid shape, or if the backend
     * implementation can report invalid index values. (Reporting about the tensor data can hinder performance, so this is implementation specific.)
     */
    abstract fun gather(axis: Int, index: Tensor): Tensor

    /**
     * Scatters the values from [source] along the specified [axis] to the indices specified in the [index] tensor, keeping other values from the
     * original tensor. (This is basically the inverse operation of [gather].)
     *
     * All three tensors must have the same number of dimensions.
     * The index must have `shape[a] <= source.shape[a]` for all axis `a`, and `shape[a] <= thisTensor.shape[a]` for all axis `a`, except for the
     * specified axis.
     * The result will have the same shape as this tensor.
     *
     * Values from the [index] tensor are converted to integer with simple casting, and invalid index value handling (outside the bounds
     * of the 0 to `shape[axis]` range) is backend implementation specific.
     *
     * Example:
     * ```
     * val tensor = Tensor.arrange(27).reshape(3, 3, 3) + 1f
     * val index = tensorOf(
     * 0, 1, 2,
     * 2, 1, 0,
     *
     * 2, 0, 1,
     * 1, 0, 2,
     *
     * 0, 2, 1,
     * 1, 2, 0
     * ).reshape(3, 2, 3)
     *
     * val source = tensor * -1f
     *
     * var res1 = tensor.copy()
     *
     * for (i in 0 until index.shape[0])
     *     for (j in 0 until index.shape[1])
     *         for (k in 0 until index.shape[2])
     *             res1[i, index[i, j, k].item().toInt(), k] = source[i, j, k]
     *
     * val res2 = tensor.scatter(axis = 1, index = index, source = source)
     *
     * println("res1:\n$res1")
     * println("res2:\n$res2")
     * ```
     * Output:
     * ```
     * res1:
     * [[[-1.0, 2.0, -6.0],
     * [4.0, -5.0, 6.0],
     * [-4.0, 8.0, -3.0]],
     * [[10.0, -14.0, 12.0],
     * [-13.0, 14.0, -12.0],
     * [-10.0, 17.0, -15.0]],
     * [[-19.0, 20.0, -24.0],
     * [-22.0, 23.0, -21.0],
     * [25.0, -23.0, 27.0]]]
     *
     * res2:
     * [[[-1.0, 2.0, -6.0],
     * [4.0, -5.0, 6.0],
     * [-4.0, 8.0, -3.0]],
     * [[10.0, -14.0, 12.0],
     * [-13.0, 14.0, -12.0],
     * [-10.0, 17.0, -15.0]],
     * [[-19.0, 20.0, -24.0],
     * [-22.0, 23.0, -21.0],
     * [25.0, -23.0, 27.0]]]
     * ```
     * @param axis the tensor axis to, the values are scattered in the result. The axis may be negative value,
     * representing `<tensor dimensions> - |<axis value>|`.
     * @param index the tensor containing the index values for the specified axis. Index values should be in the range of `0` inclusive
     * and `tensor.shape[axis]` exclusive.
     * @param source the tensor containing the values that the operation will scatter to the result.
     * @throws IllegalArgumentException if the specified axis is not in the tensor, the index, or source tensor has invalid shape, or if the
     * backend implementation can report invalid index values. (Reporting about the tensor data can hinder performance, so this is
     * implementation specific.)
     */
    abstract fun scatter(axis: Int, index: Tensor, source: Tensor): Tensor

    abstract fun indexSelect(axis: Int, index: Tensor): Tensor

    /**
     * Performs the dot product / matrix multiplication operation on tensors, based on the tensors dimensionality.
     *
     * The operation, based on the tensors dimensions is the following:
     * - If both tensors are 1D, it calculates the dot product, and returns a singleton tensor (tensor with a single value).
     * - If this tensor (left hand side) is 2D and the input tensor (right hand side) is 1D, it performs a matrix-vector
     * multiplication, and returns a vector (1D tensor).
     * - If this tensor is 1D and the input tensor is 2D, it prepends a singleton dimension to this tensor (`unsqueeze(axis = 0)`),
     * performs a matrix multiplication, after the operation, the extra dimension is removed. Returning a vector.
     * - If both tensors are 2D, it performs a matrix multiplication, and returns a matrix.
     * - If either or both tensors dimensions `>=` 3, it performs a batch matrix multiplication, by broadcasting the batch dimensions
     * (preceding the last two dimensions). If this tensor is 1D, it simply broadcasted to the shape of the input tensor, if the input
     * tensor is 1D, a singleton dimension is appended to it (`unsqueeze(axis = 1)`) and broadcasted to the shape of this tensor.
     * If both tensors have more than two dimensions, the batch dimensions must follow the implicit broadcast rule.
     *
     * **Implicit broadcasting rule:** A tensor is implicitly broadcastable to another shape, if all non-singleton dimensions
     * match in the tensor's shape and the desired shape, all missing dimensions are considered as singleton dimensions at the
     * beginning of the tensor's shape.
     *
     * Example:
     * ```
     * val t1 = Tensor.arrange(8).reshape(2, 2, 2)
     * val t2 = Tensor.arrange(12).reshape(2, 1, 2, 3)
     *
     * val res = t1 matmul t2
     *
     * println("$res shape: ${res.shape}")
     * ```
     * Output:
     * ```
     * [[[[3.0, 4.0, 5.0],
     * [9.0, 14.0, 19.0]],
     * [[15.0, 24.0, 33.0],
     * [21.0, 34.0, 47.0]]],
     * [[[9.0, 10.0, 11.0],
     * [39.0, 44.0, 49.0]],
     * [[69.0, 78.0, 87.0],
     * [99.0, 112.0, 125.0]]]] shape: [2, 2, 2, 3]
     * ```
     * @throws IllegalArgumentException if the tensor shapes are incompatible.
     */
    abstract infix fun matmul(tensor: Tensor): Tensor

    /**
     * Performs the General Matrix-matrix Multiplication
     *
     * The operation calculates `addMatrix = alpha * this @ matrix + beta * addMatrix`, where `this` tensor should be a
     * `m x k` matrix, `matrix` should be a `k x n` matrix, the `addMatrix` should be a `m x n` matrix, alpha and beta are
     * scalar scaling parameters, and the `@` operator represents the matrix multiplication operation. The result of this
     * method is the `addMatrix`.
     *
     * Note: This operation modifies the `addMatrix` inplace, but only the result of the operation captures the operation in a
     * autograd graph, so using the input instance to back propagate through will not give the same gradients as back-propagating
     * through the result of this operation.
     */
    abstract fun gemm(addMatrix: Tensor, matrix: Tensor, alpha: Float = 1f, beta: Float = 1f): Tensor

    /**
     * Performs a matrix-matrix multiplication
     *
     * This operation can only operate on 2D tensors. (For more details see [matmul] which is a more general purpose operation,
     * that uses this for matrix-matrix multiplications)
     *
     * @see matmul
     */
    abstract fun mm(matrix: Tensor): Tensor

    /**
     * General Matrix-matrix multiplication that works on tensors with 3 dimensions, where first dimension is a batch dimension.
     * (For more details see [gemm])
     *
     * @see gemm
     */
    abstract fun batchedGemm(addTensor: Tensor, tensor: Tensor, alpha: Float = 1f, beta: Float = 1f): Tensor

    /**
     * Matrix-matrix multiplication operation performed on 3D tensors, where the first dimension is a batch dimension.
     *
     * @see mm
     * @see matmul
     */
    abstract fun bmm(tensor: Tensor): Tensor

    /**
     * Performs the General Matrix-vector multiplication
     *
     * The operation calculates `addVector = alpha * this @ vector + beta * vector`, where `@` denotes the matrix-vector multiplication
     * operation, and `this` tensor is a matrix. The operation returns the `addVector` after the calculation.
     *
     * Note: This operation modifies the `addVector` inplace, but only the result of the operation captures the operation in a
     * autograd graph, so using the input instance to back propagate through will not give the same gradients as back-propagating
     * through the result of this operation.
     */
    abstract fun gemv(addVector: Tensor, vector: Tensor, alpha: Float = 1f, beta: Float = 1f): Tensor

    /**
     * Performs a matrix-vector multiplication, where `this` tensor must be a 2D tensor, and the [vector] parameter must be a 1D tensor.
     * The result is a 1D tensor (vector).
     */
    abstract fun mv(vector: Tensor): Tensor

    /**
     * Calculates the dot product of two vectors.
     * `this` tensor and the [vector] parameter tensor must be 1D tensor, and the result is a singleton tensor (a 1D tensor with a single
     * scalar component).
     *
     * Example:
     * ```
     * val tensor = tensorOf(2, 4, 6)
     * val vector = tensorOf(1, 2, 3)
     *
     * println(tensor.dot(vector))
     * ```
     * Output:
     * ```
     * [28.0]
     * ```
     */
    abstract fun dot(vector: Tensor): Tensor

    abstract fun variance(axis: Int, keepDimensions: Boolean = false, unbiased: Boolean = true): Tensor

    abstract fun std(axis: Int, keepDimensions: Boolean = false, unbiased: Boolean = true): Tensor

    /**
     *
     */
    abstract fun meanVariance(axis: Int, keepDimensions: Boolean = false, unbiased: Boolean = true): Pair<Tensor, Tensor>

    abstract fun meanStd(axis: Int, keepDimensions: Boolean = false, unbiased: Boolean = true): Pair<Tensor, Tensor>

    /**
     * Creates a view for the tensor with the same data and specified shape. (Similar to [reshape] but without copying the underlying data)
     *
     * Example:
     * ```
     * val tensor = arrange(4)
     * println(tensor)
     * val t2by2 = tensor.view(2, 2)
     * println(t2by2)
     *
     * tensor[0] = 4f
     *
     * println(t2by2)
     * ```
     * Output:
     * ```
     * [0.0, 1.0, 2.0, 3.0]
     *
     * [[0.0, 1.0],
     * [2.0, 3.0]]
     *
     * [[4.0, 1.0],
     * [2.0, 3.0]]
     * ```
     *
     * @throws IllegalArgumentException If the specified [newShape] cannot be applied to the tensor
     * @see [reshape]
     */
    abstract fun view(newShape: List<Int>): Tensor

    fun view(vararg newShape: Int) = view(newShape.asList())

    abstract infix fun lt(tensor: Tensor): Tensor
    abstract infix fun lte(tensor: Tensor): Tensor
    abstract infix fun gt(tensor: Tensor): Tensor
    abstract infix fun gte(tensor: Tensor): Tensor
    abstract infix fun eq(tensor: Tensor): Tensor
    abstract infix fun neq(tensor: Tensor): Tensor
    abstract infix fun lt(value: Float): Tensor
    abstract infix fun lte(value: Float): Tensor
    abstract infix fun gt(value: Float): Tensor
    abstract infix fun gte(value: Float): Tensor
    abstract infix fun eq(value: Float): Tensor
    abstract infix fun neq(value: Float): Tensor

    abstract fun maskedFill(mask: Tensor, value: Float): Tensor


    internal abstract fun forward()

    abstract fun backward()

    abstract fun backward(gradients: Tensor)

    abstract fun grad(): Tensor

    internal abstract fun getRawValue(): AbstractRawTensor<Any>

    abstract fun noGrad(): Tensor

    abstract fun retainGrad(): Tensor

    abstract fun asVariable(requiresGrad: Boolean = false): Tensor

    internal abstract fun platformOps(): TensorOperations<AbstractRawTensor<Any>>

    abstract fun toPlatform(platform: String): Tensor

    abstract fun release()

    abstract fun incrementRef()

    internal abstract fun serialize(): CommonSerializableTensorDescriptor

    fun <T> serializeWith(serializer: TensorSerializer<T>): T {
        return serializer.serialize(serialize())
    }


    override fun toString(): String {
        val strBuilder = StringBuilder()

        for (i in 0 until dimensions)
            strBuilder.append("[")

        val size = shape.size
        val reversedShape = shape.reversed()
        val indices = IntArray(size)
        for (it in 0 until shape.fold(1, Int::times)) {
            var i = it
            var dimensionIndex = size - 1
            for (dimension in reversedShape) {
                indices[dimensionIndex] = i % dimension
                i /= dimension
                dimensionIndex--
            }
            if (it > 0 && indices.last() == 0) {
                for (j in indices.size - 1 downTo 0) {
                    if (indices[j] == 0)
                        strBuilder.append("]")
                    else {
                        strBuilder.append(",\n")
                        for (k in 0 until indices.size - j - 1)
                            strBuilder.append("[")
                        break
                    }
                }
            }
            if (indices.last() != 0)
                strBuilder.append(", ")

            strBuilder.append(getValue(indices.toList()))
        }
        for (i in 0 until dimensions)
            strBuilder.append("]")

        return strBuilder.toString()
    }


    companion object {

        operator fun invoke(shape: List<Int>, requiresGrad: Boolean = false, init: (Int) -> Float = { 0.0f }): Tensor {
            return PlatformProvider.defaultOps().create(shape, requiresGrad, init)
        }

        operator fun invoke(vararg shape: Int, requiresGrad: Boolean = false, init: (Int) -> Float = { 0.0f }): Tensor {
            return invoke(shape.toList(), requiresGrad, init)
        }

        fun fromArray(array: FloatArray, requiresGrad: Boolean = false): Tensor {
            return PlatformProvider.defaultOps().create(listOf(array.size), requiresGrad) { array[it] }
            //TODO later this may needs to be changed to passing the actual array, to speed things up a bit
        }

        fun tensorOf(vararg values: Float, requiresGrad: Boolean = false): Tensor {
            return fromArray(values, requiresGrad)
        }

        fun tensorOf(vararg values: Int, requiresGrad: Boolean = false): Tensor {
            return fromArray(values.map { it.toFloat() }.toFloatArray(), requiresGrad)
        }

        fun tensorOf(vararg values: Double, requiresGrad: Boolean = false): Tensor {
            return fromArray(values.map { it.toFloat() }.toFloatArray(), requiresGrad)
        }

        fun arrange(value: Float, step: Float = 1.0f, start: Float = 0.0f): Tensor {
            val size = kotlin.math.round((value - start) / step).toInt()
            return invoke(listOf(size)) { start + it * step }
        }

        fun arrange(value: Int): Tensor = arrange(value.toFloat())

        fun zeros(vararg shape: Int, requiresGrad: Boolean = false): Tensor {
            return invoke(shape.toList(), requiresGrad)
        }

        fun zeros(shape: List<Int>, requiresGrad: Boolean = false): Tensor {
            return invoke(shape, requiresGrad)
        }

        fun zerosLike(other: Tensor): Tensor {
            return other.platformOps().createFillConst(other.shape, other.requiresGrad, 0f)
        }

        fun zerosLikeNoGrad(other: Tensor): Tensor {
            return other.platformOps().createFillConst(other.shape, requiresGrad = false, 0f)
        }

        fun ones(vararg shape: Int, requiresGrad: Boolean = false): Tensor {
            return PlatformProvider.defaultOps().createFillConst(shape.toList(), requiresGrad, 1f)
        }

        fun randomTensor(vararg shape: Int, requiresGrad: Boolean = false): Tensor {
            return randomTensor(shape.toList(), requiresGrad)
        }

        fun randomTensor(shape: List<Int>, requiresGrad: Boolean = false): Tensor {
            return PlatformProvider.defaultOps().createRandom(shape, requiresGrad)
        }

        fun randomLike(tensor: Tensor, requiresGrad: Boolean = false): Tensor {
            return tensor.ops.createRandom(tensor.shape, requiresGrad)
        }

        fun bernoulliDistribution(shape: List<Int>, rate: Float, requiresGrad: Boolean = false): Tensor {
            return PlatformProvider.defaultOps().createBernoulli(shape, rate, requiresGrad)
        }

        fun eye(size: Int): Tensor {
            return invoke(listOf(size, size)) {
                if (it / size == it % size) 1.0f else 0.0f
            }
        }

        fun eye(rows: Int, cols: Int): Tensor {
            return invoke(listOf(rows, cols)) {
                if (it / cols == it % cols) 1f else 0f
            }
        }

        fun oneHot(tensor: Tensor, classes: Int): Tensor {
            if (tensor.dimensions != 1) {
                throw IllegalArgumentException("One hot encoding can only be applied to 1D tensors, but got tensor with shape: ${tensor.shape}")
            }
            return invoke(listOf(tensor.shape[0], classes)) {
                val idx = it / classes
                val classIdx = it % classes
                if (classIdx == tensor.getValue(listOf(idx)).toInt()) 1f else 0f
            }.toPlatform(tensor.platform)
        }

        fun <T> deserializeWith(serializer: TensorSerializer<T>, serializedValue: T): Tensor {
            return PlatformProvider.defaultOps().fromCommonSerializable(serializer.deserialize(serializedValue))
        }

        /**
         * Calculates the elementwise exponential (e^x) function on the given tensor.
         *
         * Example:
         * ```
         * val tensor = Tensor.arrange(4).reshape(2, 2)
         * println("tensor:")
         * println(tensor)
         *
         * println("Tensor.exp(tensor):")
         * println(Tensor.exp(tensor))
         * ```
         * Output:
         * ```
         * tensor:
         * [[0.0, 1.0],
         * [2.0, 3.0]]
         *
         * Tensor.exp(tensor):
         * [[1.0, 2.7182817],
         * [7.389056, 20.085537]]
         * ```
         */
        fun exp(tensor: Tensor): Tensor = tensor.exp()
        fun log(tensor: Tensor): Tensor = tensor.log()
        fun tanh(tensor: Tensor): Tensor = tensor.tanh()
        fun sigmoid(tensor: Tensor): Tensor = tensor.sigmoid()
        fun sinh(tensor: Tensor): Tensor = tensor.sinh()
        fun cosh(tensor: Tensor): Tensor = tensor.cosh()
        fun abs(tensor: Tensor): Tensor = tensor.abs()
        fun sqrt(tensor: Tensor): Tensor = tensor.sqrt()
        fun sin(tensor: Tensor): Tensor = tensor.sin()
        fun cos(tensor: Tensor): Tensor = tensor.cos()
        fun tan(tensor: Tensor): Tensor = tensor.tan()
        fun asin(tensor: Tensor): Tensor = tensor.asin()
        fun acos(tensor: Tensor): Tensor = tensor.acos()
        fun atan(tensor: Tensor): Tensor = tensor.atan()
        fun floor(tensor: Tensor): Tensor = tensor.floor()
        fun ceil(tensor: Tensor): Tensor = tensor.ceil()
        fun round(tensor: Tensor): Tensor = tensor.round()
        fun sign(tensor: Tensor): Tensor = tensor.sign()
        fun trunc(tensor: Tensor): Tensor = tensor.trunc()
        fun rsqrt(tensor: Tensor): Tensor = tensor.rsqrt()

        fun mean(tensor: Tensor): Tensor = tensor.mean()
        fun sum(tensor: Tensor): Tensor = tensor.sum()

        fun concat(vararg tensors: Tensor, axis: Int = 0): Tensor = concat(tensors.toList(), axis)

        fun concat(tensors: List<Tensor>, axis: Int = 0): Tensor {
            if (tensors.size == 0)
                throw IllegalArgumentException("No tensors were passed to concat")

            if (tensors.size == 1)
                return tensors[0]

            return (tensors.find { it.requiresGrad } ?: tensors[0]).concat(axis, tensors) // TODO platform check must be implemented somewhere
        }

    }

}

operator fun Float.plus(tensor: Tensor): Tensor {
    return tensor.constOpTensor(this, tensor, BasicOperators.PLUS)
}

operator fun Float.minus(tensor: Tensor): Tensor {
    return tensor.constOpTensor(this, tensor, BasicOperators.MINUS)
}

operator fun Float.times(tensor: Tensor): Tensor {
    return tensor.constOpTensor(this, tensor, BasicOperators.TIMES)
}

operator fun Float.div(tensor: Tensor): Tensor {
    return tensor.constOpTensor(this, tensor, BasicOperators.DIV)
}

infix fun Float.pow(tensor: Tensor): Tensor {
    return tensor.constOpTensor(this, tensor, BasicOperators.POW)
}
