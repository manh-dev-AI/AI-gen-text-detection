const textInput = document.querySelector('#text-input');
const captureButton = document.querySelector('#capture-selection');
const analyzeButton = document.querySelector('#analyze');
const result = document.querySelector('#result');
const resultTitle = document.querySelector('#result-title');
const resultDetail = document.querySelector('#result-detail');
const resultMeta = document.querySelector('#result-meta');
const characterCount = document.querySelector('#character-count');

let modelData = null;
let stopWordsSet = null;

function setResult(title, detail, kind = 'idle', meta = '') {
  result.className = `result ${kind}`;
  resultTitle.textContent = title;
  resultDetail.textContent = detail;
  resultMeta.textContent = meta;
}

function updateCharacterCount() {
  const count = textInput.value.trim().length;
  characterCount.textContent = `${count.toLocaleString()} ký tự · Tối thiểu 20 ký tự`;
}

function setButtonsDisabled(disabled) {
  captureButton.disabled = disabled;
  analyzeButton.disabled = disabled;
}

async function getModel() {
  if (modelData) return modelData;
  try {
    const url = chrome.runtime.getURL('model_pipeline1.json');
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Không thể đọc model_pipeline1.json (HTTP ${response.status})`);
    }
    modelData = await response.json();
    stopWordsSet = new Set(modelData.stop_words || []);
    return modelData;
  } catch (error) {
    console.error('Lỗi khi nạp model:', error);
    throw new Error('Chưa thể nạp tệp mô hình offline model_pipeline1.json.');
  }
}

// Khởi tạo trước model trong bộ nhớ để phân tích tức thì
getModel().catch(() => {});

function predictOffline(text, model) {
  const lower = text.toLowerCase();
  // Regex tương đương scikit-learn r'(?u)\b\w\w+\b'
  const rawTokens = lower.match(/\b\w\w+\b/g) || [];
  const tokens = rawTokens.filter((t) => !stopWordsSet.has(t));

  if (tokens.length === 0) {
    return {
      success: false,
      error: 'Văn bản không có từ vựng tiếng Anh hợp lệ sau khi lọc stop words.',
    };
  }

  // Tạo n-grams (1, 2)
  const ngrams = [...tokens];
  for (let i = 0; i < tokens.length - 1; i++) {
    ngrams.push(`${tokens[i]} ${tokens[i + 1]}`);
  }

  // Đếm tần số xuất hiện của các từ/cụm từ có trong vocabulary
  const counts = {};
  for (const ng of ngrams) {
    if (model.weights[ng]) {
      counts[ng] = (counts[ng] || 0) + 1;
    }
  }

  const matchedTokens = Object.keys(counts);
  if (matchedTokens.length === 0) {
    return {
      success: false,
      error: 'Văn bản không khớp với bất kỳ từ vựng nào trong mô hình.',
    };
  }

  // Tính Word TF-IDF với sublinear_tf: 1.0 + ln(count) và chuẩn hóa L2
  let tfidfSqSum = 0;
  let dotProd = 0;

  for (const ng of matchedTokens) {
    const [idf, coef] = model.weights[ng];
    const tf = 1.0 + Math.log(counts[ng]);
    const val = tf * idf;
    tfidfSqSum += val * val;
    dotProd += val * coef;
  }

  const norm = Math.sqrt(tfidfSqSum);
  const margin = dotProd / norm + model.intercept;

  // Chuyển margin của Linear SVM sang thang điểm 0-1 bằng hàm sigmoid
  const clampedMargin = Math.max(-30, Math.min(30, margin));
  const score = 1.0 / (1.0 + Math.exp(-clampedMargin));
  const predictedClass = margin >= 0 ? 1 : 0;

  return {
    success: true,
    predictedClass,
    label: predictedClass === 1 ? 'AI-generated' : 'Human',
    score,
    margin,
    matchedFeatures: matchedTokens.length,
  };
}

async function captureSelection() {
  captureButton.disabled = true;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) {
      throw new Error('Không thể truy cập tab hiện tại.');
    }
    const execution = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection()?.toString().trim() || '',
    });
    const selectedText = execution[0]?.result || '';
    if (!selectedText) {
      setResult('Chưa tìm thấy phần bôi đen', 'Hãy bôi đen văn bản trên trang hiện tại, rồi thử lại.', 'warning');
      return;
    }
    textInput.value = selectedText;
    updateCharacterCount();
    setResult('Đã lấy văn bản', 'Đoạn được bôi đen đã sẵn sàng để phân tích.', 'success', `${selectedText.length.toLocaleString()} ký tự`);
  } catch (error) {
    setResult('Không thể đọc phần bôi đen', error.message || 'Lỗi khi lấy văn bản từ trang web.', 'error');
  } finally {
    captureButton.disabled = false;
  }
}

async function analyzeText() {
  const text = textInput.value.trim();
  if (text.length < 20) {
    setResult('Văn bản quá ngắn', 'Hãy nhập hoặc bôi đen ít nhất 20 ký tự để mô hình có đủ tín hiệu phân tích.', 'warning');
    return;
  }

  setButtonsDisabled(true);
  setResult('Đang phân tích…', 'Mô hình đang trích xuất Word TF-IDF và chạy Linear SVM trong trình duyệt…', 'idle');

  try {
    const model = await getModel();
    const prediction = predictOffline(text, model);

    if (!prediction.success) {
      setResult('Không thể phân tích', prediction.error, 'warning');
      return;
    }

    const aiPercentage = (prediction.score * 100).toFixed(1);
    const explanation =
      prediction.predictedClass === 1
        ? 'Mô hình nhận diện phong cách viết có nhiều nét tương đồng với văn bản do AI tạo ra.'
        : 'Mô hình nhận diện phong cách viết tự nhiên, gần với văn bản do con người viết.';

    setResult(
      prediction.label === 'AI-generated' ? 'Khả năng là văn bản AI' : 'Khả năng là con người viết (Human)',
      explanation,
      prediction.predictedClass === 1 ? 'warning' : 'success',
      `Mức điểm AI: ${aiPercentage}% · Khớp ${prediction.matchedFeatures.toLocaleString()} đặc trưng từ vựng`
    );
  } catch (error) {
    setResult('Đã xảy ra lỗi', error.message || 'Không thể thực hiện phân tích.', 'error');
  } finally {
    setButtonsDisabled(false);
  }
}

textInput.addEventListener('input', updateCharacterCount);
captureButton.addEventListener('click', captureSelection);
analyzeButton.addEventListener('click', analyzeText);
updateCharacterCount();
