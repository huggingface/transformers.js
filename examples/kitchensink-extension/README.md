# AI Context Copilot

## 🚀 Vision

AI Context Copilot is a revolutionary browser extension that transforms how users interact with web content through intelligent, context-aware AI assistance. Built with Transformers.js, it provides real-time AI capabilities directly in the browser without compromising privacy.

## ✨ Key Features

### 🧠 Smart Content Analysis
- **Text Classification**: Real-time sentiment analysis and content categorization
- **Entity Recognition**: Automatic detection of names, places, organizations, and other entities
- **Zero-Shot Classification**: Classify content into custom categories on-the-fly
- **Content Summarization**: Generate concise summaries of long articles and documents

### 👁️ Visual Intelligence
- **Image Classification**: Identify objects, scenes, and concepts in images
- **Image Captioning**: Generate descriptive captions for images
- **OCR (Optical Character Recognition)**: Extract text from images and documents
- **Visual Question Answering**: Ask questions about images and get AI responses

### 🎵 Audio Processing
- **Speech-to-Text**: Transcribe audio content in multiple languages
- **Audio Classification**: Identify sounds, music genres, and audio content
- **Zero-Shot Audio Classification**: Classify audio into custom categories
- **Text-to-Speech**: Convert text to natural-sounding speech

### ♿ Accessibility Tools
- **Screen Reader Enhancement**: Improved descriptions for images and complex content
- **Language Support**: Real-time translation and multilingual content analysis
- **Voice Navigation**: Audio-based interaction with web content
- **Visual Descriptions**: Detailed descriptions of images for visually impaired users

### 🚀 Productivity Features
- **Smart Highlights**: Automatically highlight important information
- **Context-Aware Suggestions**: Intelligent recommendations based on page content
- **Data Extraction**: Extract structured data from unstructured content
- **Cross-Site Insights**: Connect information across different websites and tabs

### 🔒 Privacy-First Design
- **Local Processing**: All AI computations happen in the browser
- **No Data Collection**: User content never leaves the device
- **Offline Capable**: Core features work without internet connection
- **Transparent Operations**: Users can see exactly what the AI is analyzing

## 🏗️ Architecture

### Core Components

1. **Background Service Worker** (`background.js`)
   - Manages AI model loading and caching
   - Handles context menus and extension lifecycle
   - Coordinates between different extension components

2. **Content Script** (`content.js`)
   - Analyzes page content in real-time
   - Provides interactive overlays and highlights
   - Handles user interactions with page elements

3. **Side Panel** (`sidepanel.html`)
   - Main AI assistant interface
   - Displays analysis results and insights
   - Provides AI chat interface for questions

4. **Popup** (`popup.html`)
   - Quick access to common features
   - Settings and preferences
   - Status and model information

5. **Options Page** (`options.html`)
   - Advanced configuration
   - Model management and updates
   - Privacy and performance settings

### AI Pipeline Integration

The extension leverages multiple Transformers.js pipelines:

- **Text Processing**
  - `text-classification` - Sentiment analysis, toxicity detection
  - `token-classification` - Named entity recognition
  - `zero-shot-classification` - Custom content categorization
  - `question-answering` - Content-based Q&A
  - `summarization` - Text summarization
  - `feature-extraction` - Semantic similarity

- **Image Processing**
  - `image-classification` - Object and scene recognition
  - `image-to-text` - Image captioning and OCR
  - `zero-shot-image-classification` - Custom image categorization
  - `object-detection` - Detect and locate objects in images

- **Audio Processing**
  - `automatic-speech-recognition` - Speech transcription
  - `audio-classification` - Sound identification
  - `zero-shot-audio-classification` - Custom audio categorization
  - `text-to-speech` - Text-to-audio synthesis

## 🎯 Target Use Cases

### For Students & Researchers
- Automatically summarize research papers and articles
- Extract key entities and concepts from academic content
- Generate questions for study materials
- Analyze sentiment in historical documents

### For Content Creators & Writers
- Analyze tone and sentiment of written content
- Extract key topics and themes from competitor content
- Generate alt-text for images automatically
- Transcribe video content for captions

### For Accessibility Users
- Enhanced screen reader support with AI-generated descriptions
- Real-time transcription of audio content
- Visual content explanation for images and complex layouts
- Voice-controlled browsing assistance

### For Business Professionals
- Sentiment analysis of customer feedback and reviews
- Automatic data extraction from forms and documents
- Content categorization for research and analysis
- Cross-language content understanding

### For General Web Users
- Understanding complex or technical content
- Quick summaries of long articles
- Image and video content explanation
- Enhanced search and discovery

## 🚀 Getting Started

### Prerequisites
- Chrome/Chromium browser version 116+
- 4GB+ RAM recommended for optimal performance
- Modern GPU recommended for faster processing

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build the extension:
   ```bash
   npm run build
   ```
4. Load the extension in Chrome:
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `dist` folder

### Development

```bash
# Start development server with hot reload
npm run dev

# Build for production
npm run build
```

## 🔧 Configuration

### Model Selection
The extension automatically downloads and caches AI models optimized for browser use. Models are sourced from the WebML collection on Hugging Face, ensuring compatibility and performance.

### Performance Tuning
- **Memory Management**: Models are loaded on-demand and cached intelligently
- **Background Processing**: Heavy computations run in background threads
- **Progressive Enhancement**: Features gracefully degrade on lower-end devices

## 🛣️ Roadmap

### Phase 1: Core Intelligence (Current)
- ✅ Text analysis and classification
- ✅ Image understanding and captioning
- ✅ Audio processing and transcription
- ✅ Basic accessibility features

### Phase 2: Advanced Features
- 🔄 Multi-modal analysis (combining text, image, audio)
- 🔄 Conversation memory and context
- 🔄 Custom model fine-tuning
- 🔄 Advanced productivity workflows

### Phase 3: Integration & Ecosystem
- 📋 API for other extensions
- 📋 Integration with popular web apps
- 📋 Custom model marketplace
- 📋 Advanced analytics and insights

## 🤝 Contributing

We welcome contributions! This project demonstrates the cutting-edge possibilities of browser-based AI and we're excited to push the boundaries together.

### Areas for Contribution
- New AI pipeline integrations
- Performance optimizations
- Accessibility improvements
- UI/UX enhancements
- Documentation and examples

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers.js** - For making browser-based AI possible
- **WebML Community** - For optimized model collections
- **Open Source AI Community** - For advancing accessible AI technology

---

*Built with ❤️ and 🤖 for the future of intelligent web browsing*
