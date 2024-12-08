# FLUX.1 AI Image Generation Interface 🎨

A powerful multi-model AI image generation interface using Hugging Face's Inference API with advanced prompt engineering and parallel processing capabilities.

## 🌟 Features

- Multiple specialized AI models for different art styles
- Advanced prompt enhancement system with 5 detail levels
- Parallel image generation with fallback models
- Token rotation for reliable API access
- Real-time prompt preview
- Safety filters and content controls
- High-quality image generation (up to 1024x1024)

## 🚀 Quick Start

1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Hugging Face API token:
   - Create a file named `.env` in the project root
   - Add your token(s):
     ```
     HF_TOKEN=your_token_here
     # For multiple tokens (recommended):
     HF_TOKEN_1=your_first_token
     HF_TOKEN_2=your_second_token
     HF_TOKEN_3=your_third_token
     ```
   - Get your tokens from: https://huggingface.co/settings/tokens

## 💻 Usage

Run the application:
```bash
python apiflux1.py
```

The interface will be available at: http://localhost:7860

## 🎯 Enhancement Levels

1. **Level 1 (Minimal)**
   - Basic quality improvements
   - Light detail enhancement

2. **Level 2 (Moderate)**
   - Added focus and intricate details
   - Model-specific enhancements

3. **Level 3 (Detailed)**
   - Professional quality terms
   - Advanced composition

4. **Level 4 (Very Detailed)**
   - 4K/8K quality terms
   - Advanced lighting and effects

5. **Level 5 (Maximum)**
   - Studio quality rendering
   - Maximum detail and effects

## 🔒 Security

- Never commit your `.env` file
- Use token rotation for reliability
- Enable safety filters for public use
- Monitor token usage

## 🛠️ Troubleshooting

1. **API Errors**
   - Check your token validity
   - Ensure you have enough API quota
   - Try using multiple tokens

2. **Image Quality**
   - Increase enhancement level
   - Use model-specific prompts
   - Adjust quality parameters

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.
