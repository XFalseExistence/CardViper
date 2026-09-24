package com.cardviper.app.ui.imagetest

import androidx.annotation.MainThread
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.cardviper.app.vision.*
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ImageRecognitionViewModel(
    private val loader: ImageSourceLoader,
    private val engine: ImageRecognitionEngine,
) : ViewModel() {
    private val mutableState = MutableStateFlow(ImageRecognitionUiState())
    val uiState = mutableState.asStateFlow()
    private var analysis: Job? = null
    private val stateLock = Any()
    private var requestId = 0L

    @MainThread
    fun analyze(source: String) {
        analysis?.cancel()
        val request = synchronized(stateLock) {
            requestId += 1
            mutableState.value = ImageRecognitionUiState(
                phase = ImageTestPhase.LOADING_IMAGE, source = source, statusText = "LOADING IMAGE",
            )
            requestId
        }
        analysis = viewModelScope.launch {
            try {
                val image = loader.load(source)
                ensureActive()
                publish(request) { it.copy(phase = ImageTestPhase.DETECTING, image = image, statusText = "DETECTING CARDS") }
                val result = withContext(Dispatchers.Default) {
                    engine.recognize(image) { progress ->
                        publish(request) {
                            it.copy(
                                phase = when (progress.stage) {
                                    RecognitionStage.DETECTING -> ImageTestPhase.DETECTING
                                    RecognitionStage.CLASSIFYING -> ImageTestPhase.CLASSIFYING
                                },
                                progress = progress,
                                statusText = when (progress.stage) {
                                    RecognitionStage.DETECTING -> "DETECTING CARDS"
                                    RecognitionStage.CLASSIFYING -> "CLASSIFYING ${progress.completed}/${progress.total}"
                                },
                            )
                        }
                    }
                }
                ensureActive()
                publish(request) {
                    it.copy(
                        phase = if (result.observations.isEmpty()) ImageTestPhase.NO_CARDS else ImageTestPhase.COMPLETE,
                        result = result,
                        progress = null,
                        statusText = if (result.observations.isEmpty()) "NO CARDS DETECTED" else "ANALYSIS COMPLETE",
                    )
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (failure: Exception) {
                publish(request) {
                    it.copy(phase = ImageTestPhase.ERROR, result = null, progress = null, statusText = when (failure) {
                        is ImageLoadException -> "COULD NOT OPEN IMAGE"
                        is ImageDecodeException -> "COULD NOT DECODE IMAGE"
                        is VisionModelUnavailableException -> "VISION MODEL NOT INSTALLED"
                        is DetectorInferenceException -> "CARD DETECTOR FAILED"
                        is ClassifierInferenceException -> "CARD CLASSIFIER FAILED"
                        else -> "IMAGE ANALYSIS FAILED"
                    })
                }
            }
        }
    }

    @MainThread
    fun reset() {
        analysis?.cancel()
        analysis = null
        synchronized(stateLock) {
            requestId += 1
            mutableState.value = ImageRecognitionUiState()
        }
    }

    // Inference progress can arrive from a worker thread, even after cancellation.
    // Check and publish atomically so old work cannot overwrite a newer selection.
    private fun publish(request: Long, update: (ImageRecognitionUiState) -> ImageRecognitionUiState) {
        synchronized(stateLock) {
            if (request == requestId) mutableState.value = update(mutableState.value)
        }
    }

    override fun onCleared() {
        synchronized(stateLock) { requestId += 1 }
        super.onCleared()
    }
}
