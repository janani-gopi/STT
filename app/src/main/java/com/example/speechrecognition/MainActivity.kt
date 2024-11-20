package com.example.speechrecognition

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.ImageButton
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private val PERMISSION_REQUEST_CODE = 123
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var tflite: Interpreter? = null
    private lateinit var transcriptionTextView: TextView
    private lateinit var micButton: ImageButton
    private var recordingJob: Job? = null

    // Audio configuration
    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(
        SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT
    )

    // Whisper model configuration
    private val AUDIO_SEGMENT_LENGTH = 30 * SAMPLE_RATE // 30 seconds of audio
    private val PROCESSING_STEP = 1 * SAMPLE_RATE // Process every 1 second

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        transcriptionTextView = findViewById(R.id.transcriptionText)
        micButton = findViewById(R.id.micButton)
        
        setupMicButton()

        // Initialize TFLite regardless of permissions
        initializeTFLite()
        
        // Check permissions
        checkAudioPermission()
    }

    private fun setupMicButton() {
        micButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
                micButton.setImageResource(R.drawable.ic_mic_off) // Make sure to add these icons to your drawable folder
            } else {
                if (checkAudioPermission()) {
                    startRecording()
                    micButton.setImageResource(R.drawable.ic_mic_on)
                }
            }
        }
    }

    private fun checkAudioPermission(): Boolean {
        return if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSION_REQUEST_CODE
            )
            false
        } else {
            true
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Only initialize TFLite if not already initialized
                if (tflite == null) {
                    initializeTFLite()
                }
            } else {
                Log.e("MainActivity", "Audio permission denied")
                transcriptionTextView.text = "Audio permission required"
                micButton.isEnabled = false
            }
        }
    }


    private fun initializeTFLite() {
    try {
        // Open the model file from the assets folder
        val assetFileDescriptor = assets.openFd("whisper-tiny.tflite")
        val fileDescriptor = assetFileDescriptor.fileDescriptor
        val fileChannel = assetFileDescriptor.createInputStream().channel
        val startOffset = assetFileDescriptor.startOffset
        val size = assetFileDescriptor.length
        val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, size)

        // Initialize the TFLite interpreter
        tflite = Interpreter(mappedByteBuffer)

    } catch (e: Exception) {
        Log.e("MainActivity", "Error loading TFLite model", e)
        transcriptionTextView.text = "Model loading failed"
        micButton.isEnabled = false
    }

}


    private fun startRecording() {
        if (tflite == null || 
            ActivityCompat.checkSelfPermission(
                this, 
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }

        try {
            val audioBuffer = ShortArray(BUFFER_SIZE)
            val processingBuffer = FloatArray(AUDIO_SEGMENT_LENGTH)
            var bufferIndex = 0

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE
            )

            isRecording = true
            audioRecord?.startRecording()

            recordingJob = CoroutineScope(Dispatchers.Default).launch {
                while (isRecording) {
                    val readSize = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: 0

                    for (i in 0 until readSize) {
                        processingBuffer[bufferIndex] = audioBuffer[i] / 32768.0f
                        bufferIndex++

                        if (bufferIndex >= PROCESSING_STEP) {
                            processAudio(processingBuffer)

                            processingBuffer.copyInto(
                                processingBuffer,
                                0,
                                PROCESSING_STEP,
                                AUDIO_SEGMENT_LENGTH
                            )
                            bufferIndex = AUDIO_SEGMENT_LENGTH - PROCESSING_STEP
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Recording error", e)
            transcriptionTextView.text = "Recording failed"
            stopRecording()
        }
    }

    private fun stopRecording() {
        isRecording = false
        recordingJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    private fun processAudio(audioData: FloatArray) {
        Log.d("AudioData", "Processing audio buffer: ${audioData.joinToString()}")
        try {            
            val inputBuffer = ByteBuffer.allocateDirect(audioData.size * 4)
                .order(ByteOrder.nativeOrder())

            for (value in audioData) {
                inputBuffer.putFloat(value)
            }

            val outputBuffer = ByteBuffer.allocateDirect(1024)
                .order(ByteOrder.nativeOrder())

            tflite?.let { interpreter ->
                val inputs = arrayOf<Any>(inputBuffer)
                val outputs = mapOf(0 to outputBuffer)

                interpreter.runForMultipleInputsOutputs(inputs, outputs)

                val transcription = decodeOutput(outputBuffer)

                runOnUiThread {
                    updateTranscriptionUI(transcription)
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Audio processing error", e)
            runOnUiThread {
                transcriptionTextView.text = "Processing error"
            }
        }
    }

    private fun decodeOutput(output: ByteBuffer): String {
        return try {
            output.rewind()
            val bytes = ByteArray(output.remaining())
            output.get(bytes)
            String(bytes).trim()
        } catch (e: Exception) {
            Log.e("MainActivity", "Output decoding error", e)
            "Transcription error"
        }
    }

    private fun updateTranscriptionUI(text: String) {
        transcriptionTextView.text = text
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRecording()
        tflite?.close()
    }
}