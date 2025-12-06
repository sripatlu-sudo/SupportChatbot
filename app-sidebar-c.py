st.markdown(
    """
    <style>

    /* -----------------------------------------
       COLLEGE ADMISSIONS BROCHURE THEME
       Clean • Elegant • Academic
       ----------------------------------------- */

    body, .stApp {
        background-image: url("https://images.unsplash.com/photo-1503676260728-1c00da094a0b");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: "Georgia", "Merriweather", serif;
        color: #1D1A1A;
    }

    /* Overlay for readability */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.60);
        z-index: -1;
    }

    /* Main Title */
    .college-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-top: -10px;
        color: #00274C; /* University Navy Blue */
        font-family: "Merriweather", serif;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.15);
    }

    .college-subtitle {
        text-align: center;
        color: #4A4A4A;
        font-size: 18px;
        margin-bottom: 30px;
        font-family: "Georgia";
    }

    /* Chat Bubbles — Brochure Style */
    .chat-bubble-user {
        padding: 16px;
        background: #F7F9FC;
        border-left: 5px solid #00274C;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        font-size: 16px;
    }

    .chat-bubble-bot {
        padding: 16px;
        background: #FFFFFF;
        border-left: 5px solid #FFCB05; /* University Gold */
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-size: 16px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #FFF4D9;
        border-left: 4px solid #FFCB05;
        padding: 12px;
        border-radius: 6px;
        color: #5A4A3B;
        font-size: 15px;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #A2A2A2;
        background: rgba(255,255,255,0.75);
        font-family: "Georgia";
    }

    .stCheckbox label {
        color: #00274C !important;
        font-weight: 600;
    }

    /* Buttons — Classic Academic Look */
    .stButton>button {
        background-color: #00274C !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.3rem !important;
        border: none !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: "Merriweather", serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
    }

    .stButton>button:hover {
        background-color: #013a73 !important;
    }

    /* History Sidebar */
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: #00274C;
        font-family: "Merriweather", serif;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #00274C;
        padding: 12px 0;
        text-align: center;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.15);
    }

    .footer a {
        font-size: 16px;
        color: #FFCB05;
        font-family: "Georgia";
        font-weight: 600;
        text-decoration: none;
    }

    </style>

    <div class="college-title">🎓 College Picker Chatbot</div>
    <div class="college-subtitle">Your Admissions Information Assistant</div>
    """,
    unsafe_allow_html=True
)
