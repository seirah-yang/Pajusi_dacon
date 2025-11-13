let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const chatBox = document.getElementById("chat-box");
const micBtn = document.getElementById("mic-btn");
const ttsPlayer = document.getElementById("tts-player");

function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = cls;
  div.innerHTML = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById("user-input");
  const userMsg = input.value.trim();
  if (!userMsg) return;

  addMessage("👤 " + userMsg, "user-msg");
  input.value = "";

  const formData = new FormData();
  formData.append("query", userMsg);

  const res = await fetch("/paju/query", { method: "POST", body: formData });
  const data = await res.json();

  const botText = data.answer || "답변을 가져오지 못했습니다.";
  addMessage("🤖 " + botText, "bot-msg");

  if (data.tts_path) {
    playTTS(data.tts_path);
  }
}

async function toggleMic() {
  if (!isRecording) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = handleAudioStop;

    mediaRecorder.start();
    micBtn.classList.add("recording");
    micBtn.textContent = "⏹️";
    isRecording = true;
  } else {
    mediaRecorder.stop();
    micBtn.classList.remove("recording");
    micBtn.textContent = "🎤";
    isRecording = false;
  }
}

async function handleAudioStop() {
  const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
  const formData = new FormData();
  formData.append("user_id", "sora");
  formData.append("audio", audioBlob, "input.wav");

  addMessage("🎙️ 음성 인식 중...", "bot-msg");

  const res = await fetch("/paju/voice-chat", { method: "POST", body: formData });
  const data = await res.json();

  if (data.recognized_text)
    addMessage("👤 (음성) " + data.recognized_text, "user-msg");

  addMessage("🤖 " + (data.answer || "답변을 불러올 수 없습니다."), "bot-msg");

  if (data.tts_path) {
    playTTS(data.tts_path);
  }
}

function playTTS(ttsPath) {
  if (!ttsPath) return;
  ttsPlayer.src = ttsPath;
  ttsPlayer.style.display = "block";
  ttsPlayer.play().catch(() => console.warn("TTS 재생 실패"));
}

document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("mic-btn").addEventListener("click", toggleMic);
document.getElementById("user-input").addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});
