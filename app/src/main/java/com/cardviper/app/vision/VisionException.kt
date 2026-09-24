package com.cardviper.app.vision

class VisionModelUnavailableException(message: String) : Exception(message)

class DetectorInferenceException(message: String, cause: Throwable? = null) : Exception(message, cause)

class ClassifierInferenceException(message: String, cause: Throwable? = null) : Exception(message, cause)
