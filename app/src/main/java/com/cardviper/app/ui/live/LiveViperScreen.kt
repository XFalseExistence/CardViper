package com.cardviper.app.ui.live

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.blackjack.ResolvedCard
import com.cardviper.app.model.PlayingCard
import com.cardviper.app.session.SessionSnapshot
import com.cardviper.app.session.SessionState
import com.cardviper.app.ui.review.CardCorrectionSheet

private enum class LiveDialog { SHOE_INFO, VISION_DEBUG, NEW_SHOE, END_SHOE }

@Composable
fun LiveViperScreen(
    snapshot: SessionSnapshot,
    recentCards: List<ResolvedCard>,
    busy: Boolean,
    showRecentCards: Boolean,
    onBack: () -> Unit,
    onSwitchStrategy: (CountStrategyId) -> Unit,
    onAddCard: (PlayingCard) -> Unit,
    onCorrectCard: (String, PlayingCard) -> Unit,
    onInvalidateCard: (String) -> Unit,
    onUndo: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onReview: () -> Unit,
    onNewShoe: () -> Unit,
    onEndShoe: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val session = snapshot.session
    var permissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { permissionGranted = it }
    LaunchedEffect(Unit) { if (!permissionGranted) permissionLauncher.launch(Manifest.permission.CAMERA) }
    DisposableEffect(context, lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                permissionGranted = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    var frozen by rememberSaveable(session.sessionId) { mutableStateOf(false) }
    var cameraStatus by remember(session.sessionId) { mutableStateOf(CameraStatus.STARTING) }
    var cameraAttempt by remember(session.sessionId) { mutableIntStateOf(0) }
    var menuOpen by remember { mutableStateOf(false) }
    var manualAddOpen by remember { mutableStateOf(false) }
    var correction by remember { mutableStateOf<ResolvedCard?>(null) }
    var dialog by remember { mutableStateOf<LiveDialog?>(null) }
    val cameraActive = permissionGranted && session.state == SessionState.ACTIVE && !frozen

    Column(modifier = Modifier.fillMaxSize().background(Color(0xFF050807))) {
        BoxWithConstraints(modifier = Modifier.fillMaxWidth().weight(1f).background(Color.Black)) {
            if (cameraActive) key(cameraAttempt) { CameraPreview(modifier = Modifier.fillMaxSize()) { cameraStatus = it } }
            Row(modifier = Modifier.align(Alignment.TopStart).fillMaxWidth().padding(8.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                TextButton(onClick = onBack) { Text("‹ BACK") }
                Box {
                    TextButton(onClick = { menuOpen = true }) { Text("MORE ⋮") }
                    DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                        DropdownMenuItem(
                            text = { Text(if (session.state == SessionState.PAUSED) "Resume shoe" else "Pause shoe") },
                            enabled = !busy && session.state != SessionState.ENDED,
                            onClick = { menuOpen = false; if (session.state == SessionState.PAUSED) onResume() else onPause() },
                        )
                        DropdownMenuItem(text = { Text(if (frozen) "Unfreeze preview" else "Freeze preview") }, enabled = !busy && session.state == SessionState.ACTIVE,
                            onClick = { menuOpen = false; frozen = !frozen })
                        DropdownMenuItem(text = { Text("Vision debug") }, onClick = { menuOpen = false; dialog = LiveDialog.VISION_DEBUG })
                        DropdownMenuItem(text = { Text("Shoe info") }, onClick = { menuOpen = false; dialog = LiveDialog.SHOE_INFO })
                        DropdownMenuItem(text = { Text("New shoe") }, enabled = !busy, onClick = { menuOpen = false; dialog = LiveDialog.NEW_SHOE })
                        DropdownMenuItem(text = { Text("End shoe") }, enabled = !busy, onClick = { menuOpen = false; dialog = LiveDialog.END_SHOE })
                    }
                }
            }
            Column(
                modifier = Modifier.align(Alignment.Center).fillMaxWidth()
                    .heightIn(max = (maxHeight - 88.dp).coerceAtLeast(48.dp)).verticalScroll(rememberScrollState()).padding(horizontal = 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                when {
                    !permissionGranted -> {
                        Text("Camera permission needed for live preview")
                        Text("Your shoe and manual controls remain available.")
                        Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) { Text("RETRY PERMISSION") }
                        TextButton(onClick = {
                            context.startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                                data = Uri.fromParts("package", context.packageName, null)
                            })
                        }) { Text("OPEN APP SETTINGS") }
                    }
                    session.state == SessionState.PAUSED -> Text("Shoe paused")
                    session.state == SessionState.READY -> Text("Shoe ready")
                    session.state == SessionState.ENDED -> Text("Shoe ended")
                    frozen -> Text("Preview frozen")
                    cameraStatus == CameraStatus.ERROR -> {
                        Text("Camera unavailable")
                        Button(onClick = { cameraStatus = CameraStatus.STARTING; cameraAttempt += 1 }) { Text("RETRY CAMERA") }
                    }
                    cameraStatus == CameraStatus.STARTING -> Text("Starting camera…")
                    cameraStatus == CameraStatus.PAUSED -> Text("Camera paused")
                }
            }
            Text(
                when {
                    !permissionGranted -> "CAMERA PERMISSION REQUIRED · 0 TRACKS"
                    session.state != SessionState.ACTIVE -> "CAMERA PAUSED · 0 TRACKS"
                    frozen -> "CAMERA FROZEN · 0 TRACKS"
                    cameraStatus == CameraStatus.ACTIVE -> "CAMERA ACTIVE · 0 TRACKS · RECOGNITION OFF"
                    cameraStatus == CameraStatus.ERROR -> "CAMERA ERROR · 0 TRACKS"
                    else -> "CAMERA STARTING · 0 TRACKS"
                },
                modifier = Modifier.align(Alignment.BottomCenter).padding(8.dp).background(Color(0xD9000000)).padding(horizontal = 10.dp, vertical = 6.dp),
                color = Color(0xFFE5BF70),
            )
        }
        Box(modifier = Modifier.fillMaxWidth().heightIn(max = 360.dp)) {
            LiveHud(
                snapshot = snapshot,
                recentCards = recentCards,
                busy = busy,
                showRecentCards = showRecentCards,
                onSwitchStrategy = onSwitchStrategy,
                onAddCard = { manualAddOpen = true },
                onUndo = onUndo,
                onReview = onReview,
                onCorrectCard = { correction = it },
            )
        }
    }

    if (manualAddOpen) ManualAddCardDialog(strategy = session.countStrategy, busy = busy, onDismiss = { manualAddOpen = false }, onAddCard = onAddCard)
    correction?.let { selected ->
        CardCorrectionSheet(
            strategy = session.countStrategy,
            current = selected.card,
            busy = busy,
            onDismiss = { correction = null },
            onCorrect = { onCorrectCard(selected.effectiveEventId, it) },
            onRemove = { onInvalidateCard(selected.effectiveEventId) },
        )
    }
    when (dialog) {
        LiveDialog.SHOE_INFO -> LiveInfoDialog("Shoe info", "${session.nominalDecks} decks · ${session.countStrategy.name.replace('_', ' ')} · ${session.startMode.name.replace('_', ' ')}\nState: ${session.state.name.lowercase()} · Cards: ${snapshot.cardsSeen}") { dialog = null }
        LiveDialog.VISION_DEBUG -> LiveInfoDialog("Vision debug", "Recognition is not enabled in this version.\nActive tracks: 0 · Locked tracks: 0") { dialog = null }
        LiveDialog.NEW_SHOE -> LiveConfirmDialog("New shoe?", "Start a new shoe setup. The current shoe stays saved.", "NEW SHOE", { dialog = null }) { dialog = null; onNewShoe() }
        LiveDialog.END_SHOE -> LiveConfirmDialog("End shoe?", "This ends the current shoe and saves its history.", "END SHOE", { dialog = null }) { dialog = null; onEndShoe() }
        null -> Unit
    }
}

@Composable
private fun LiveInfoDialog(title: String, message: String, onDismiss: () -> Unit) {
    AlertDialog(onDismissRequest = onDismiss, title = { Text(title) }, text = { Text(message) },
        confirmButton = { TextButton(onClick = onDismiss) { Text("CLOSE") } })
}

@Composable
private fun LiveConfirmDialog(title: String, message: String, confirm: String, onDismiss: () -> Unit, onConfirm: () -> Unit) {
    AlertDialog(onDismissRequest = onDismiss, title = { Text(title) }, text = { Text(message) },
        confirmButton = { TextButton(onClick = onConfirm) { Text(confirm) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("CANCEL") } })
}
