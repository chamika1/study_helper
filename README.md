# A/L Physics Learning Assistant

[![A/L Physics Learning Assistant](https://img.shields.io/badge/Visit%20App-Live-brightgreen)](https://al-physics.techbloglk.xyz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-orange)](https://www.buymeacoffee.com/jLufrxRaO1)

An AI-powered educational tool designed specifically for Sri Lankan Advanced Level students studying Physics. This application leverages artificial intelligence to provide personalized learning experiences, practice questions, exam tips, study plans, and concise notes.

## ✨ Features

* **🧠 Interactive Learning**: Step-by-step explanations of complex physics concepts tailored to the A/L syllabus.
* **📝 Practice Questions**: Generate unlimited practice questions with varying difficulty levels to test understanding.
* **💡 Exam Tips**: Access strategic advice and common pitfalls for excelling in A/L Physics examinations.
* **📅 Study Plans**: Receive customized study schedules based on individual progress and time availability.
* **✍️ Short Notes**: Get concise, easy-to-understand revision materials for quick review sessions.
* **🌐 Bilingual Support**: Seamlessly switch between English and Sinhala for a comfortable learning experience.

## 🛠️ Technologies Used

* **Backend**: Python with Flask framework
* **Frontend**: HTML, CSS, JavaScript
* **Styling**: Bootstrap 5 for responsive design
* **AI Engine**: Google Gemini API for intelligent content generation
* **Deployment**: Hosted on cloud infrastructure for accessibility

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites

* Python 3.8+
* pip (Python package installer)
* Git

### Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/chamika1/study_helper.git](https://github.com/chamika1/study_helper.git)
    cd study_helper
    ```

2.  **Create a virtual environment and activate it:**
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add your Google Gemini API key:
    ```env
    GEMINI_API_KEY=YOUR_API_KEY_HERE
    ```
    *Replace `YOUR_API_KEY_HERE` with your actual API key.*

5.  **Run the application:**
    ```bash
    flask run
    ```
    *Alternatively, you can use `python app.py`.*

6.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000` or `http://localhost:5000`.

## 📁 Project Structure

study_helper/│├── app.py              # Main Flask application logic├── physics.txt         # Data file containing physics topics and units├── requirements.txt    # Python package dependencies├── .env                # Environment variables (contains API key - not committed)├── templates/          # HTML templates for the web interface│   ├── index.html      # Main page template│   └── ...             # Other necessary templates│├── static/             # Static files (CSS, JavaScript, Images)│   ├── css/│   │   └── style.css   # Custom CSS styles│   ├── js/│   │   └── script.js   # Custom JavaScript│   └── images/         # Application images/icons│├── venv/               # Virtual environment directory (not committed)└── README.md           # This file
## 📖 Usage

1.  Navigate to the application in your browser.
2.  Select the desired Physics **Unit** and **Topic** from the dropdown menus.
3.  Choose your preferred **Language** (English or Sinhala).
4.  Select a **Learning Mode** (e.g., Learn Step-by-Step, Practice Questions, Exam Tips, Study Plan, Short Notes).
5.  Click the "Generate" or "Submit" button.
6.  Interact with the AI assistant's response to deepen your understanding or get more practice.

## 🤝 Contributing

Contributions are highly welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeatureName`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeatureName`).
5.  Open a Pull Request.

Please ensure your code adheres to standard Python coding conventions (PEP 8).

## ❤️ Support

If you find this project helpful and would like to support its development, you can:

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/jLufrxRaO1)

Your support is greatly appreciated!

## 🔗 Connect with the Developer

* **Chamika:**
    * [Personal Website](https://itzmechami.wuaze.com/)
    * [GitHub](https://github.com/chamika1)
* **Project & Community:**
    * [Tech Blog](https://techbloglk.xyz/)
    * [Telegram Group](https://t.me/study_helper_physics)
    * [Facebook Page](https://web.facebook.com/people/Techbloglk/61575108633834/)

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

* A heartfelt thank you to all the dedicated Sri Lankan physics teachers who inspire and educate students.
* The Flask and Python communities for their excellent frameworks and documentation.
* Google for providing access to the powerful Gemini AI API.
