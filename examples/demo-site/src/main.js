

import Chart from 'chart.js/auto';
import Prism from 'prismjs';

// Import code and styles for supported languages
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-clike';
import 'prismjs/themes/prism.css'

import './theme.css';
import './style.css';

// Initialise worker
const worker = new Worker(new URL('./worker.js', import.meta.url), {
  type: 'module',
});


// Define elements
const TASK_SELECTOR = document.getElementById('task');

let searchParams = new URLSearchParams(location.search);
let defaultDemo = searchParams.get('demo');
if (defaultDemo) {
  TASK_SELECTOR.value = defaultDemo;
}

// translation inputs
const LANGUAGE_FROM = document.getElementById('language-from');
const LANGUAGE_TO = document.getElementById('language-to');
const INPUT_TEXTBOX = document.getElementById('input-textbox');
const OUTPUT_TEXTBOX = document.getElementById('output-textbox');

// text generation inputs
const TEXT_GENERATION_TEXTBOX = document.getElementById('text-generation-textbox');

const TASKS = document.getElementsByClassName('task-settings')
const PROGRESS = document.getElementById('progress');
const PROGRESS_BARS = document.getElementById('progress-bars');

const GENERATE_BUTTON = document.getElementById('generate');

const MLM_INPUT_TEXTBOX = document.getElementById('mlm-input-textbox');
const MLM_OUTPUT_TEXTBOX = document.getElementById('mlm-output-textbox');

const SC_INPUT_TEXTBOX = document.getElementById('sc-input-textbox');
const SC_OUTPUT_CANVAS = document.getElementById('sc-canvas');

const TC_INPUT_TEXTBOX = document.getElementById('tc-input-textbox');
const TC_OUTPUT = document.getElementById('tc-output');

const QA_CONTEXT_TEXTBOX = document.getElementById('qa-context-textbox');
const QA_QUESTION_TEXTBOX = document.getElementById('qa-question-textbox');
const QA_ANSWER_TEXTBOX = document.getElementById('qa-answer-textbox');

const SUMMARIZATION_INPUT_TEXTBOX = document.getElementById('summarization-input-textbox');
const SUMMARIZATION_OUTPUT_TEXTBOX = document.getElementById('summarization-output-textbox');

const SPEECH2TEXT_SELECT = document.getElementById('audio-select');
const SPEECH2TEXT_INPUT = document.getElementById('audio-file');
const SPEECH2TEXT_AUDIO = document.getElementById('audio-player');
const SPEECH2TEXT_OUTPUT_TEXTBOX = document.getElementById('speech2text-output-textbox');


const TEXT2IMAGE_SELECT = document.getElementById('image-select');
const TEXT2IMAGE_INPUT = document.getElementById('image-file');
const TEXT2IMAGE_IMG = document.getElementById('image-viewer');
const TEXT2IMAGE_OUTPUT_TEXTBOX = document.getElementById('image2text-output-textbox');

const IMAGE_CLASSIFICATION_SELECT = document.getElementById('ic-select');
const IMAGE_CLASSIFICATION_INPUT = document.getElementById('ic-file');
const IMAGE_CLASSIFICATION_IMG = document.getElementById('ic-viewer');
const IMAGE_CLASSIFICATION_OUTPUT_CANVAS = document.getElementById('ic-canvas');

const CODE_COMPLETION_CONTAINER = document.getElementById('code-completion-container');


const ZSIC_SELECT = document.getElementById('zsic-select');
const ZSIC_INPUT = document.getElementById('zsic-file');
const ZSIC_CLASSES = document.getElementById('zsic-classes');
const ZSIC_IMG = document.getElementById('zsic-viewer');
const ZSIC_OUTPUT_CANVAS = document.getElementById('zsic-canvas');


const OD_SELECT = document.getElementById('od-select');
const OD_INPUT = document.getElementById('od-file');
const OD_IMG = document.getElementById('od-viewer');
const OD_OUTPUT_OVERLAY = document.getElementById('od-overlay');
const OD_OUTPUT_CANVAS = document.getElementById('od-canvas');


const ZSC_INPUT_TEXTBOX = document.getElementById('zsc-input-textbox');
const ZSC_CLASSES = document.getElementById('zsc-classes');
const ZSC_OUTPUT_CANVAS = document.getElementById('zsc-canvas');


const DEFAULT_GREEDY_PARAMS = {
  max_new_tokens: 50,
  num_beams: 1,
  temperature: 1,
  top_k: 0,
  do_sample: false
}

const TASK_DEFAULT_PARAMS = {
  'translation': DEFAULT_GREEDY_PARAMS,
  'text-generation': {
    max_new_tokens: 100,
    num_beams: 1,
    temperature: 1,
    top_k: 20,
    do_sample: true
  },
  'code-completion': DEFAULT_GREEDY_PARAMS,
  'masked-language-modelling': {
    topk: 5 // number of samples
  },
  'sequence-classification': {},
  'token-classification': {},
  'zero-shot-classification': {
    multi_label: false
  },
  'question-answering': {},
  'summarization': {
    max_new_tokens: 50,
    num_beams: 2,
    temperature: 1,
    top_k: 0,
    do_sample: false
  },
  'automatic-speech-recognition': DEFAULT_GREEDY_PARAMS,
  'image-to-text': DEFAULT_GREEDY_PARAMS,
  'image-classification': {},
  'zero-shot-image-classification': {},
  'object-detection': {},
};

[
  [SPEECH2TEXT_SELECT, SPEECH2TEXT_INPUT, SPEECH2TEXT_AUDIO],
  [TEXT2IMAGE_SELECT, TEXT2IMAGE_INPUT, TEXT2IMAGE_IMG],
  [IMAGE_CLASSIFICATION_SELECT, IMAGE_CLASSIFICATION_INPUT, IMAGE_CLASSIFICATION_IMG],
  [ZSIC_SELECT, ZSIC_INPUT, ZSIC_IMG],
  [OD_SELECT, OD_INPUT, OD_IMG],
].forEach(x => {
  let [select, input, media] = x;

  select.addEventListener('input', (e) => {
    if (select.options[select.selectedIndex].hasAttribute('show-custom')) {
      input.style.display = 'block';
    } else {
      input.style.display = 'none';

      media.src = select.value
    }
  })

  input.addEventListener("change", () => {
    const file = input.files[0];
    const url = URL.createObjectURL(file);
    media.src = url;
  });
});

const NER_TAGS = {
  // tag: [textColour, backgroundColour, tagColour]
  'ORG': ['#115E59', '#CCFBF1', '#14B8A6'],
  'PER': ['#9D174D', '#FCE7F3', '#EC4899'],
  'LOC': ['#86198F', '#FAE8FF', '#D946EF'],
}


// Predefined list of unique colours
const COLOURS = [
  '255, 99, 132',
  '54, 162, 235',
  '255, 206, 86',
  '75, 192, 192',
  '153, 102, 255',
  '255, 159, 64',
]



OD_SELECT.addEventListener('change', () => {
  // Clear overlay and chart data on change
  OD_OUTPUT_OVERLAY.innerHTML = '';

  const chart = CHARTS[OD_OUTPUT_CANVAS.id];
  chart.data = structuredClone(DEFAULT_DATA);
  chart.update();
});

OD_OUTPUT_OVERLAY.addEventListener('mousemove', (e) => {
  let rects = OD_OUTPUT_OVERLAY.querySelectorAll('rect')

  let colours = [];
  let borderColours = [];


  rects.forEach((rect, i) => {
    let colour = COLOURS[i % COLOURS.length];

    // Display if hovering over background (tagName === 'svg')
    let toDisplay = e.target.tagName !== 'rect';
    if (!toDisplay) {
      // Perform additional check
      let bb = rect.getBoundingClientRect()

      // Check if box intersects with current mouse positition
      toDisplay = e.clientX >= bb.left && e.clientX <= bb.right && e.clientY >= bb.top && e.clientY <= bb.bottom
    }

    if (toDisplay) {
      // Set back to original
      rect.style.fillOpacity = 0.1;
      rect.style.opacity = 1;
      colours.push(`rgba(${colour}, 0.5)`);
      borderColours.push(`rgba(${colour}, 1)`);
    } else {
      // Hovering over a rect, so set all other rects to 0 opacity
      rect.style.fillOpacity = 0;
      rect.style.opacity = 0;
      colours.push(`rgba(${colour}, 0.05)`);
      borderColours.push(`rgba(${colour}, 0.5)`);
    }

  })

  const chart = CHARTS['od-canvas'];
  chart.data.datasets[0].backgroundColor = colours;
  chart.data.datasets[0].borderColor = borderColours;
  chart.update();
})

function updateParams(task) {
  let params = TASK_DEFAULT_PARAMS[task]
  if (!params) return;

  for (let [key, value] of Object.entries(params)) {
    let element = document.querySelector(`.generation-option[param-name="${key}"]`)
    if (!element) continue;
    element.value = value;
  }
}

// Parameters
const GENERATION_OPTIONS = document.getElementsByClassName('generation-option');


const CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  indexAxis: 'y',
  scales: {
    y: {
      beginAtZero: true,
    },
    x: {
      min: 0,
      max: 1,
    }
  },
  plugins: {
    legend: {
      display: false
    },
  },
  layout: {
    padding: {
      bottom: -5,
    }
  },
};


// Initialise all code blocks
const CODE_BLOCKS = {};
[...document.querySelectorAll('.code-container')].forEach(element => {

  // Guide to add editable code block:
  // https://codepen.io/WebCoder49/pen/dyNyraq
  // https://css-tricks.com/creating-an-editable-textarea-that-supports-syntax-highlighted-code/

  const CODE_HIGHLIGHT = element.querySelector('pre');
  const CODE_HIGHLIGHT_CONTENT = element.querySelector('code');
  const CODE_COMPLETION_TEXTBOX = element.querySelector('textarea');

  let sync_scroll = () => {
    /* Scroll result to scroll coords of event - sync with textarea */
    CODE_HIGHLIGHT.scrollTop = CODE_COMPLETION_TEXTBOX.scrollTop;
    CODE_HIGHLIGHT.scrollLeft = CODE_COMPLETION_TEXTBOX.scrollLeft;
  }
  let update = (text) => {
    // Handle final newlines (see article)
    if (text[text.length - 1] == "\n") {
      text += " ";
    }
    // Update code
    CODE_HIGHLIGHT_CONTENT.innerHTML = escapeHtml(text);

    // Syntax Highlight
    Prism.highlightElement(CODE_HIGHLIGHT_CONTENT);
  }

  // Update code function
  let updateCode = (text) => {
    update(text);
    sync_scroll();
  };

  CODE_BLOCKS[element.id] = {
    update: (text) => {
      CODE_COMPLETION_TEXTBOX.value = text;
      updateCode(text);

      // When updating, set scroll to bottom
      // https://stackoverflow.com/a/9170709
      CODE_COMPLETION_TEXTBOX.scrollTop = CODE_COMPLETION_TEXTBOX.scrollHeight;
    },
    text: () => CODE_COMPLETION_TEXTBOX.value
  };

  CODE_COMPLETION_TEXTBOX.oninput = () => updateCode(CODE_COMPLETION_TEXTBOX.value);

  CODE_COMPLETION_TEXTBOX.onscroll = sync_scroll;
  CODE_COMPLETION_TEXTBOX.onkeydown = (event) => {
    let code = CODE_COMPLETION_TEXTBOX.value;
    if (event.key == "Tab") {
      /* Tab key pressed */
      event.preventDefault(); // stop normal
      let before_tab = code.slice(0, CODE_COMPLETION_TEXTBOX.selectionStart); // text before tab
      let after_tab = code.slice(CODE_COMPLETION_TEXTBOX.selectionEnd, CODE_COMPLETION_TEXTBOX.value.length); // text after tab
      let cursor_pos = CODE_COMPLETION_TEXTBOX.selectionStart + 1; // where cursor moves after tab - moving forward by 1 char to after tab
      CODE_COMPLETION_TEXTBOX.value = before_tab + "\t" + after_tab; // add tab char
      // move cursor
      CODE_COMPLETION_TEXTBOX.selectionStart = cursor_pos;
      CODE_COMPLETION_TEXTBOX.selectionEnd = cursor_pos;
      update(CODE_COMPLETION_TEXTBOX.value); // Update text to include indent
    }
  };

});

const DEFAULT_DATA = {
  labels: ['label', 'label', 'label', 'label', 'label'],
  datasets: [{
    borderWidth: 1
  }]
}

const CHARTS = {
  'sc-canvas': new Chart(SC_OUTPUT_CANVAS, {
    type: 'bar',
    data: {
      labels: ['5 stars', '4 stars', '3 stars', '2 stars', '1 star'],
      datasets: [{
        borderWidth: 1
      }]
    },
    options: CHART_OPTIONS,

  }),
  'ic-canvas': new Chart(IMAGE_CLASSIFICATION_OUTPUT_CANVAS, {
    type: 'bar',
    data: structuredClone(DEFAULT_DATA),
    options: CHART_OPTIONS
  }),

  'zsic-canvas': new Chart(ZSIC_OUTPUT_CANVAS, {
    type: 'bar',
    data: {
      labels: ['football', 'airport', 'animals'],
      datasets: [{
        borderWidth: 1
      }]
    },
    options: CHART_OPTIONS
  }),

  'od-canvas': new Chart(OD_OUTPUT_CANVAS, {
    type: 'bar',
    data: structuredClone(DEFAULT_DATA),
    options: CHART_OPTIONS
  }),

  'zsc-canvas': new Chart(ZSC_OUTPUT_CANVAS, {
    type: 'bar',
    data: {
      labels: ['urgent', 'not urgent', 'phone', 'tablet', 'microwave'],
      datasets: [{
        borderWidth: 1
      }]
    },
    options: CHART_OPTIONS
  }),
};


[
  [ZSIC_CLASSES, ZSIC_OUTPUT_CANVAS],
  [ZSC_CLASSES, ZSC_OUTPUT_CANVAS],
].forEach(x => {
  let [input, chart] = x;

  input.addEventListener('input', () => {
    // Update labels of graph
    let chartToUpdate = CHARTS[chart.id];

    chartToUpdate.data.labels = getZSClasses(input);
    chartToUpdate.data.datasets[0].data = new Array(chartToUpdate.data.labels.length).fill(0);
    chartToUpdate.update();
  })
});


function getZSClasses(elem) {
  // Get zero-shot classes from input element
  return elem.value.split(/\s*,+\s*/g).filter(x => x);
}

function updateVisibility() {

  // Set default parameters for task
  updateParams(TASK_SELECTOR.value);

  for (let element of TASKS) {
    if (element.getAttribute('task').split(',').includes(TASK_SELECTOR.value)) {
      element.style.display = 'block';
    } else {
      element.style.display = 'none';
    }
  }
}
updateVisibility();

// Add event listeners
TASK_SELECTOR.addEventListener('input', updateVisibility);

function parseValue(value, type) {
  switch (type) {
    case 'number':
      return Number(value);
    case 'bool':
      return value === 'true'
    default:
      return value
  }
}

function isVisible(e) {
  // https://stackoverflow.com/a/38873788
  return !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
}

GENERATE_BUTTON.addEventListener('click', async (e) => {
  // Set and pass generation settings to web worker
  let data = {
    task: TASK_SELECTOR.value,
    generation: Object.fromEntries([...GENERATION_OPTIONS]
      .filter(isVisible) // Only use parameters that are visible on screen
      .map(x => {
        let value = parseValue(x.value, x.getAttribute('datatype'));
        return [x.getAttribute('param-name'), value]
      }))
  };
  switch (TASK_SELECTOR.value) {
    case 'translation':
      data.languageFrom = LANGUAGE_FROM.value
      data.languageTo = LANGUAGE_TO.value
      data.text = INPUT_TEXTBOX.value
      data.elementIdToUpdate = OUTPUT_TEXTBOX.id
      break;

    case 'text-generation':
      data.text = TEXT_GENERATION_TEXTBOX.value
      data.elementIdToUpdate = TEXT_GENERATION_TEXTBOX.id
      break;

    case 'code-completion':
      data.text = CODE_BLOCKS[CODE_COMPLETION_CONTAINER.id].text();
      data.elementIdToUpdate = CODE_COMPLETION_CONTAINER.id
      data.targetType = 'code'
      break;

    case 'masked-language-modelling':
      data.text = MLM_INPUT_TEXTBOX.value
      data.elementIdToUpdate = MLM_OUTPUT_TEXTBOX.id
      break;

    case 'sequence-classification':
      data.text = SC_INPUT_TEXTBOX.value
      data.elementIdToUpdate = SC_OUTPUT_CANVAS.id
      data.targetType = 'chart'
      break;

    case 'token-classification':
      data.text = TC_INPUT_TEXTBOX.value
      data.elementIdToUpdate = TC_OUTPUT.id
      data.targetType = 'tokens'
      break;

    case 'zero-shot-classification':
      data.text = ZSC_INPUT_TEXTBOX.value
      data.classes = getZSClasses(ZSC_CLASSES);
      data.elementIdToUpdate = ZSC_OUTPUT_CANVAS.id
      data.targetType = 'chart'
      data.updateLabels = true
      break;

    case 'question-answering':
      data.context = QA_CONTEXT_TEXTBOX.value
      data.question = QA_QUESTION_TEXTBOX.value
      data.elementIdToUpdate = QA_ANSWER_TEXTBOX.id
      break;

    case 'summarization':
      data.text = SUMMARIZATION_INPUT_TEXTBOX.value
      data.elementIdToUpdate = SUMMARIZATION_OUTPUT_TEXTBOX.id
      break;

    case 'automatic-speech-recognition':
      const sampling_rate = 16000;
      const audioCTX = new AudioContext({ sampleRate: sampling_rate })

      const response = await (await fetch(SPEECH2TEXT_AUDIO.currentSrc)).arrayBuffer()
      const decoded = await audioCTX.decodeAudioData(response)

      data.audio = decoded.getChannelData(0);
      data.elementIdToUpdate = SPEECH2TEXT_OUTPUT_TEXTBOX.id
      break;

    case 'image-to-text':
      data.image = getImageDataFromImage(TEXT2IMAGE_IMG)
      data.elementIdToUpdate = TEXT2IMAGE_OUTPUT_TEXTBOX.id
      break;

    case 'image-classification':
      data.image = getImageDataFromImage(IMAGE_CLASSIFICATION_IMG)
      data.elementIdToUpdate = IMAGE_CLASSIFICATION_OUTPUT_CANVAS.id
      data.targetType = 'chart'
      data.updateLabels = true
      break;


    case 'zero-shot-image-classification':
      data.image = getImageDataFromImage(ZSIC_IMG)
      data.classes = getZSClasses(ZSIC_CLASSES);
      data.elementIdToUpdate = ZSIC_OUTPUT_CANVAS.id
      data.targetType = 'chart'
      data.updateLabels = true
      break;

    case 'object-detection':
      data.image = getImageDataFromImage(OD_IMG)
      data.targetType = 'overlay'
      data.chartId = OD_OUTPUT_CANVAS.id
      data.elementIdToUpdate = OD_OUTPUT_OVERLAY.id
      break;

    default:
      return;
  }

  worker.postMessage(data);
});

// Handle result returned by the web worker

worker.addEventListener('message', (event) => {
  const message = event.data;

  switch (message.type) {
    case 'download': // for session creation

      if (message.data.status === 'initiate') {
        PROGRESS.style.display = 'block';

        // create progress bar
        PROGRESS_BARS.appendChild(htmlToElement(`
					<div class="progress w-100" model="${message.data.name}" file="${message.data.file}">
						<div class="progress-bar" role="progressbar"></div>
					</div>
				`));

      } else {
        let bar = PROGRESS_BARS.querySelector(`.progress[model="${message.data.name}"][file="${message.data.file}"]> .progress-bar`)

        switch (message.data.status) {
          case 'progress':
            // update existing bar
            bar.style.width = message.data.progress.toFixed(2) + '%';
            bar.textContent = `${message.data.file} (${formatBytes(message.data.loaded)} / ${formatBytes(message.data.total)})`;
            break;

          case 'done':
            // Remove the progress bar
            bar.parentElement.remove();
            break;

          case 'ready':
            // Pipeline is ready - hide container
            PROGRESS.style.display = 'none';
            PROGRESS_BARS.innerHTML = '';
            break;
        }
      }

      break;
    case 'update': // for generation
      let target = message.target;
      let elem = document.getElementById(target);

      switch (message.targetType) {
        case 'code':
          CODE_BLOCKS[target].update(message.data);
          break;
        default: // is textbox
          elem.value += message.data
          break;
      }

      break;

    case 'complete':
      switch (message.targetType) {
        case 'chart':
          const chartToUpdate = CHARTS[message.target];

          let chartData = chartToUpdate.data.datasets[0].data;

          if (message.updateLabels) {
            for (let i = 0; i < message.data.length; ++i) {
              let item = message.data[i];
              chartData[i] = item.score;
              chartToUpdate.data.labels[i] = item.label;
            }
          } else {
            // set data, ensuring labels align correctly
            for (let item of message.data) {
              chartData[
                chartToUpdate.data.labels.indexOf(item.label)
              ] = item.score
            }
          }

          chartToUpdate.update(); // update the chart
          break;

        case 'tokens':
          let target = document.getElementById(message.target);
          target.innerHTML = '';

          let tokens = message.data;

          for (let token of tokens) {
            let elem;
            if (token.type === 'O') {
              elem = document.createTextNode(token.text);
            } else {
              let [textColour, backgroundColour, tagColour] = NER_TAGS[token.type];
              elem = htmlToElement(`<span class="ner-container" style="background-color: ${backgroundColour}; color: ${textColour};">${token.text}<span class="ner-tag" style="background-color: ${tagColour}; color: ${backgroundColour};">${token.type}</span></span>`);
            }
            target.appendChild(elem);

          }
          break;

        case 'overlay':
          let parent = document.getElementById(message.target);

          // Clear previous output, just in case
          parent.innerHTML = '';

          let viewbox = parent.viewBox.baseVal;

          let colours = [];
          let borderColours = [];

          let items = message.data;
          for (let i = 0; i < items.length; ++i) {
            const box = items[i].box;

            let svgns = "http://www.w3.org/2000/svg";
            let rect = document.createElementNS(svgns, 'rect');

            rect.setAttribute('x', viewbox.width * box.xmin);
            rect.setAttribute('y', viewbox.height * box.ymin);
            rect.setAttribute('width', viewbox.width * (box.xmax - box.xmin));
            rect.setAttribute('height', viewbox.height * (box.ymax - box.ymin));

            const colour = COLOURS[i % COLOURS.length];
            rect.style.stroke = rect.style.fill = `rgba(${colour}, 1)`;

            colours.push(`rgba(${colour}, 0.5)`);
            borderColours.push(`rgba(${colour}, 1)`);
            parent.appendChild(rect);
          }

          // Update chart label and data
          const chart = CHARTS[message.chartId];
          chart.data.labels = items.map(x => x.label);
          chart.data.datasets[0] = {
            data: items.map(x => x.score),
            backgroundColor: colours,
            borderColor: borderColours
          };
          chart.update()
          break;
        default: // is text
          document.getElementById(message.target).value = message.data
          break;
      }
      break;
    default:
      break;
  }
});

// Utility functions

function escapeHtml(unsafe) {
  return unsafe.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}


function htmlToElement(html) {
  // https://stackoverflow.com/a/35385518
  let template = document.createElement('template');
  html = html.trim(); // Never return a text node of whitespace as the result
  template.innerHTML = html;
  return template.content.firstChild;
}

function formatBytes(bytes, decimals = 0) {
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  if (bytes === 0) return "0 Bytes";
  const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1000)), 10);
  const rounded = (bytes / Math.pow(1000, i)).toFixed(decimals);
  return rounded + " " + sizes[i];
}

function getImageDataFromImage(original) {

  // Helper function to get image data from image element
  const canvas = document.createElement('canvas');
  canvas.width = original.naturalWidth;
  canvas.height = original.naturalHeight;

  const ctx = canvas.getContext('2d');
  // TODO play around with ctx options?
  // ctx.patternQuality = 'bilinear';
  // ctx.quality = 'bilinear';
  // ctx.antialias = 'default';
  // ctx.imageSmoothingQuality = 'high';

  ctx.drawImage(original, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL();
}
