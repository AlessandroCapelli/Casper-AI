# Casper AI - LLM Experimentation Platform

A comprehensive platform for experimenting with Large Language Models (LLMs), featuring a chat interface for model interaction and fine-tuning capabilities. This project serves as a research and development environment for exploring different aspects of LLM behavior, performance, and customization.

## 🎯 Project Overview

Casper AI is designed to be a flexible and extensible platform for LLM research and development. It provides:

- A modern Angular 19 frontend for intuitive model interaction
- A Python backend with Flask for robust API handling
- A modular LLaMA model implementation for experimentation
- Fine-tuning capabilities for model customization
- Real-time chat interface for model evaluation
- Comprehensive logging and monitoring tools

## 🔬 Research Features

### Model Experimentation
- **Architecture Variations**: Experiment with different model configurations
- **Parameter Tuning**: Adjust model parameters for optimal performance
- **Performance Metrics**: Track and analyze model behavior
- **Comparative Analysis**: Compare different model configurations

### Fine-tuning Capabilities
- **Dataset Integration**: Support for custom training datasets
- **Training Pipeline**: Streamlined fine-tuning process
- **Model Checkpointing**: Save and load model states
- **Hyperparameter Optimization**: Tools for tuning training parameters

### Evaluation Tools
- **Interactive Testing**: Real-time model evaluation through chat
- **Performance Metrics**: Track response quality and latency
- **Behavior Analysis**: Monitor model outputs and patterns
- **Error Analysis**: Identify and analyze failure cases

## 🛠️ Technical Architecture

### Frontend (Angular 19)
- Modern, responsive chat interface
- Real-time message updates
- Model configuration controls
- Performance monitoring dashboard
- Dataset management interface

### Backend (Python/Flask)
- RESTful API endpoints
- Model inference pipeline
- Fine-tuning infrastructure
- Data preprocessing tools
- Logging and monitoring system

### LLM Implementation
- LLaMA model architecture
- Customizable model parameters
- Fine-tuning support
- Inference optimization
- Memory-efficient processing

## 📋 Prerequisites

- Node.js (v18 or higher)
- Python 3.8 or higher
- PyTorch
- Angular CLI
- CUDA-capable GPU (recommended for fine-tuning)

## 🚀 Getting Started

### Backend Setup

1. Navigate to the project root directory
2. Install Python dependencies:
   ```bash
   pip install flask flask-cors torch transformers datasets
   ```
3. Start the backend server:
   ```bash
   python backend.py
   ```
   The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the UI directory:
   ```bash
   cd UI
   ```
2. Install Node.js dependencies:
   ```bash
   yarn install
   ```
3. Start the development server:
   ```bash
   yarn start
   ```
   The frontend will run on `http://localhost:4200`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 TODO List

The following features and improvements are planned for future development:

- **Markdown Support**: Implement markdown rendering for chat messages
- **Enhanced Formatting**: Improve the visual presentation and formatting of the chat interface
- **Model Fine-tuning**: Expand the fine-tuning capabilities with additional parameters and datasets
- **RAG Integration**: Add Retrieval-Augmented Generation to enhance response quality with external knowledge
- **Automatic Message Scrolling**: Implement smooth automatic scrolling for new messages
- **Chat History Support**: Enable the model to reference previous conversations for context
- **Performance Optimizations**: Improve inference speed and memory usage
- **Multi-Model Support**: Add capability to switch between different model architectures
- **Export/Import Functionality**: Allow saving and loading of chat sessions
- **User Authentication**: Implement secure login and user profiles
- **Clear Chat Button**: Add a button to clear the chat history and start fresh conversations
