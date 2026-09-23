package com.cardviper.app.ui.live

import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraState
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LiveData
import androidx.lifecycle.Observer
import androidx.lifecycle.compose.LocalLifecycleOwner

enum class CameraStatus { STARTING, ACTIVE, PAUSED, ERROR }

/** Rear camera preview only; no frames are captured or analyzed. */
@Composable
fun CameraPreview(modifier: Modifier = Modifier, onStatus: (CameraStatus) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentOnStatus = rememberUpdatedState(onStatus)
    val previewView = remember(context) {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    AndroidView(factory = { previewView }, modifier = modifier)

    DisposableEffect(context, lifecycleOwner, previewView) {
        var disposed = false
        var failed = false
        var hasStreamed = false
        var bound = false
        var provider: ProcessCameraProvider? = null
        var cameraState: LiveData<CameraState>? = null
        var preview: Preview? = null

        fun report(status: CameraStatus) {
            if (!disposed && (!failed || status == CameraStatus.ERROR)) {
                currentOnStatus.value(status)
            }
        }

        fun reportStreamState(streamState: PreviewView.StreamState?) {
            if (bound && streamState == PreviewView.StreamState.STREAMING &&
                lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)
            ) {
                hasStreamed = true
                report(CameraStatus.ACTIVE)
            } else {
                report(if (hasStreamed || !lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) {
                    CameraStatus.PAUSED
                } else {
                    CameraStatus.STARTING
                })
            }
        }

        val streamObserver = Observer<PreviewView.StreamState> { reportStreamState(it) }
        val cameraObserver = Observer<CameraState> { state ->
            if (state.error != null) {
                failed = true
                report(CameraStatus.ERROR)
            } else if (failed && bound) {
                failed = false
                reportStreamState(previewView.previewStreamState.value)
            }
        }
        val lifecycleObserver = LifecycleEventObserver { _, _ ->
            reportStreamState(previewView.previewStreamState.value)
        }

        report(CameraStatus.STARTING)
        previewView.previewStreamState.observeForever(streamObserver)
        lifecycleOwner.lifecycle.addObserver(lifecycleObserver)

        try {
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener({
                if (!disposed) {
                    try {
                        val ready = future.get()
                        if (!disposed) {
                            val useCase = Preview.Builder().build().also {
                                it.setSurfaceProvider(previewView.surfaceProvider)
                            }
                            provider = ready
                            preview = useCase
                            val camera = ready.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                useCase,
                            )
                            bound = true
                            cameraState = camera.cameraInfo.cameraState.also {
                                it.observeForever(cameraObserver)
                            }
                            reportStreamState(previewView.previewStreamState.value)
                        }
                    } catch (_: Exception) {
                        failed = true
                        report(CameraStatus.ERROR)
                        cameraState?.removeObserver(cameraObserver)
                        preview?.let { useCase -> runCatching { provider?.unbind(useCase) } }
                        bound = false
                    }
                }
            }, ContextCompat.getMainExecutor(context))
        } catch (_: Exception) {
            failed = true
            report(CameraStatus.ERROR)
        }

        onDispose {
            disposed = true
            lifecycleOwner.lifecycle.removeObserver(lifecycleObserver)
            previewView.previewStreamState.removeObserver(streamObserver)
            cameraState?.removeObserver(cameraObserver)
            preview?.let { useCase -> runCatching { provider?.unbind(useCase) } }
        }
    }
}
