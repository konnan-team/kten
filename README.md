# KTen - Tensor library for Kotlin

KTen is a tensor library for Kotlin with automatic differentiation capabilities. 

KTen currently has two backend implementations: 
- `kten-jvm` is a JVM only implementation.
- `kten-opencl` is an OpenCL based implementation.

## Build

To build the project simply run:

```
mvn clean package -DskipTests
```

Or if you'd like to add it to the local maven repository, run:

```
mvn clean install -DskipTests
```

## Usage

To use the library you have to include the API and at least the default implementation (`kten-jvm`) into your project, e.g. using Maven:

```xml
<dependencies>
    <!-- other dependencies -->
    <dependency>
        <groupId>eu.redbean.kten</groupId>
        <artifactId>kten-api</artifactId>
        <version>0.1.0-SNAPSHOT</version>
    </dependency>
    <dependency>
        <groupId>eu.redbean.kten.backend</groupId>
        <artifactId>kten-jvm</artifactId>
        <version>0.1.0-SNAPSHOT</version>
    </dependency>
</dependencies>
```

With the dependencies added you can start to use the `Tensor` class in your Kotlin code (which provides all the tensor operation functionality and also the base class for all tensor implementations):

```kotlin
val tensor = Tensor.tensorOf(
    1, 2, 3,
    3, 2, 1
).reshape(2, 3)
// creates a 2x3 tensor with the specified values converted to Float
```

### Tensor operations

Most of the tensor operations are defined in the `Tensor` class itself, documentation for their usage is available on the methods in KDoc form.
(Documentation is far from complete yet, but it will come shortly :) )

### Automatic differentiation

KTen supports automatic differentiation, by recording the operations on the tensors, and gradients for the variables are calculated with a backward pass, using the `backward()` method. Variables that require gradients must be specified explicitly in the tensor creation method with the `requiresGrad` parameter, or by using the `asVariable(requiresGrad = true)` method. Gradients can be accessed after the backward pass, with the `grad()` method.

Example:

```kotlin
val tensor = Tensor.tensorOf(
    1, 2, 3,
    3, 2, 1
).reshape(2, 3).asVariable(requiresGrad = true)

Tensor.sum(tensor pow 2).backward()

println(tensor.grad())
```

Which will print:

```
[[2.0, 4.0, 6.0],
[6.0, 4.0, 2.0]]
```

### How it works

KTen is mostly inspired by [PyTorch](https://pytorch.org), and [NumPy](https://numpy.org), so it has a similar API as those frameworks. 

Tensors in KTen store their values in `RawTensor`s, implemented platform specifically (JVM or OpenCL for now), these `RawTensor`s store the scalar components of the tensor and provide a simplified tensor operation set, for the high level tensor implementations. 

KTen only supports Float data type for the tensor values.

Automatic differentiation is implemented by returning `Function` descendant instances from all `Tensor` methods, where gradient calculation is required. These `Function`s implement the forward and backward passes for the individual operations (with various other housekeeping tasks), and for the backward pass the calculations are applied in the chain-rule. (Basically the callstack takes care of it.)

Gradients for `Variable`s are aggregated in a `RawTensor` instance. (For this reason, currently only first derivatives can be calculated.)

KTen doesn't support inplace operations yet, such as `timesAssign`, `plusAssign` etc. With the only exception of `set(index, value)` for obvious reasons. Also there are no views (like the ones in PyTorch), so `get` and `reshape` operations will always copy the tensor data. (This probably will change in the future, to improve performance.)

### Tests

There is a `kten-testing` module which is included in all backend modules with test scope, and can be run with any backend. Tests in this module use the high-level `Tensor` api to check regression issues. Platform specific tests are implemented in the backend modules. 
(Currently the tests cannot run automatically, because of the OpenCL backend uses the device name in the platform selection, but this will be fixed soon.)

# Contribution

All contributions are appreciated. If you find any bugs or would like to suggest a new feature, please open an [issue](https://github.com/konnan-team/kten/issues). 

If you'd like to contribute bug-fixes, or new features, please open an issue first, if there aren't any, describing the bug or feature, and mention the issue in your PR. If there is a obvious way to demonstrate the bug-fix or feature in a test, please also implement one in the `kten-testing` module, or if it is related to one of the backend modules, then in that module's tests. 
