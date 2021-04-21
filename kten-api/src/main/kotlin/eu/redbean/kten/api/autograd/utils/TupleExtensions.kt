package eu.redbean.kten.api.autograd

inline fun <T, A>Pair<A, A>.map(mapping: (A) -> T): Pair<T, T> {
    return mapping(this.first) to mapping(this.second)
}

inline fun <T>Pair<T, T>.applyUnpack(block: (T, T) -> Unit) = block(this.first, this.second)


inline fun <T, A>Triple<A, A, A>.map(mapping: (A) -> T): Triple<T, T, T> {
    return Triple(mapping(this.first), mapping(this.second), mapping(this.third))
}

inline fun <T>Triple<T, T, T>.applyUnpack(block: (T, T, T) -> Unit) = block(this.first, this.second, this.third)
