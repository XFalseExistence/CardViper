package com.cardviper.app.ui.imagetest

import android.graphics.Paint
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.cardviper.app.vision.CardLabelCodec
import com.cardviper.app.vision.ImageCardObservation
import com.cardviper.app.vision.VisionImage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

private val UncertainYellow = Color(0xFFFFD54F)

@Composable
fun ImageTestScreen(state: ImageRecognitionUiState, onImageSelected: (String) -> Unit, onBack: () -> Unit) {
    BackHandler(onBack = onBack)
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) onImageSelected(uri.toString())
    }
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 20.dp),
        contentPadding = PaddingValues(vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        item {
            TextButton(onClick = onBack) { Text("BACK") }
            Text("IMAGE TEST", style = MaterialTheme.typography.headlineMedium)
            Text("Local, offline processing. Image results never change your active shoe.",
                style = MaterialTheme.typography.bodySmall)
            Text("Trained automatic card recognition is not active. Vision models are not installed yet.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        item {
            Button(onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                modifier = Modifier.fillMaxWidth()) {
                Text(if (state.source == null) "CHOOSE IMAGE" else "CHOOSE ANOTHER IMAGE")
            }
        }
        item {
            Text(state.statusText, style = MaterialTheme.typography.titleMedium,
                color = if (state.phase == ImageTestPhase.ERROR) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary)
            if (state.phase in listOf(ImageTestPhase.LOADING_IMAGE, ImageTestPhase.DETECTING, ImageTestPhase.CLASSIFYING)) {
                LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp))
            }
        }
        state.image?.let { image ->
            item {
                // A new source gets its own derived bitmap resource, never the previous image's pixels.
                key(image) { FittedImage(image, state.result?.observations.orEmpty()) }
            }
        }
        state.result?.let { result ->
            item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("IMAGE-LOCAL DIAGNOSTIC", style = MaterialTheme.typography.labelLarge)
                        Text("Detected observations: ${result.observations.size}")
                        Text("KO Δ: ${result.koDelta}")
                        Text("KISS III Δ: ${result.kiss3Delta}")
                        Text("Uncertain: ${result.uncertainCount}")
                        if (result.skippedCandidates > 0) Text("Skipped candidates: ${result.skippedCandidates}")
                    }
                }
            }
            itemsIndexed(result.observations) { index, observation ->
                Text("${index + 1}. ${observation.label()}",
                    color = if (observation.uncertain) UncertainYellow else MaterialTheme.colorScheme.onSurface)
            }
        }
    }
}

@Composable
private fun FittedImage(image: VisionImage, observations: List<ImageCardObservation>) {
    // Only a derived rendering resource is kept here; all analysis phases/results come from the ViewModel.
    val rendered by produceState<Result<ImageBitmap>?>(null, image) {
        value = try {
            Result.success(withContext(Dispatchers.Default) { image.toDisplayBitmap().asImageBitmap() })
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Exception) {
            Result.failure(failure)
        } catch (failure: OutOfMemoryError) {
            Result.failure(failure)
        }
    }
    val bitmap = rendered?.getOrNull()
    if (bitmap == null) {
        if (rendered?.isFailure == true) Text("COULD NOT DISPLAY IMAGE") else CircularProgressIndicator()
        return
    }
    val primary = MaterialTheme.colorScheme.primary
    val paint = remember { Paint(Paint.ANTI_ALIAS_FLAG) }
    Canvas(Modifier.fillMaxWidth().height(320.dp).background(Color.Black)
        .semantics { contentDescription = "Selected image with ${observations.size} card observations" }) {
        if (size.width <= 0f || size.height <= 0f) return@Canvas
        val fit = FitCenterTransform.create(image.width.toFloat(), image.height.toFloat(), size.width, size.height)
        withTransform({
            translate(fit.offsetX, fit.offsetY)
            scale(fit.scale, fit.scale, pivot = Offset.Zero)
        }) {
            clipRect(0f, 0f, image.width.toFloat(), image.height.toFloat()) {
                drawImage(bitmap)
                observations.forEach { observation ->
                    val box = observation.candidate
                    val left = box.x.coerceIn(0f, image.width.toFloat())
                    val top = box.y.coerceIn(0f, image.height.toFloat())
                    val right = (box.x.toDouble() + box.width).coerceIn(0.0, image.width.toDouble()).toFloat()
                    val bottom = (box.y.toDouble() + box.height).coerceIn(0.0, image.height.toDouble()).toFloat()
                    if (right > left && bottom > top) drawRect(
                        color = if (observation.uncertain) UncertainYellow else primary,
                        topLeft = Offset(left, top), size = Size(right - left, bottom - top),
                        style = Stroke(width = 2.dp.toPx() / fit.scale),
                    )
                }
            }
        }
        // Text stays legible at screen scale; positions use the same tested source transform.
        paint.textSize = 12.sp.toPx()
        observations.forEachIndexed { index, observation ->
            val text = "${index + 1}. ${observation.label()}"
            val labelWidth = paint.measureText(text)
            val x = fit.mapX(observation.candidate.x.coerceIn(0f, image.width.toFloat()))
                .coerceIn(0f, (size.width - labelWidth).coerceAtLeast(0f))
            val baseline = fit.mapY(observation.candidate.y.coerceIn(0f, image.height.toFloat()))
                .coerceIn(-paint.fontMetrics.ascent, size.height - paint.fontMetrics.descent)
            drawRect(Color.Black.copy(alpha = 0.85f), Offset(x, baseline + paint.fontMetrics.ascent),
                Size(labelWidth, paint.fontMetrics.descent - paint.fontMetrics.ascent))
            paint.color = (if (observation.uncertain) UncertainYellow else primary).toArgb()
            drawContext.canvas.nativeCanvas.drawText(text, x, baseline, paint)
        }
    }
}

private fun ImageCardObservation.label(): String =
    "${CardLabelCodec.encode(identity)} · ${(classifierConfidence * 100).roundToInt()}%" +
        if (uncertain) " · UNCERTAIN" else ""
