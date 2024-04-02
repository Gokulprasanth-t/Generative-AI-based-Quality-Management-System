import streamlit as st
import assemblyai as aai
import matplotlib.pyplot as plt

# Set your API key
aai.settings.api_key = "a8d25ef3200e4508b2f5e87449ef61a0"

def audio_processing(filename):
    filepath = "C:/Users/gokul/Downloads/ML Task/Call Data Sample/" + filename
    
    # Configuring the model for sentiment analysis
    cfg = aai.TranscriptionConfig(sentiment_analysis=True)
    
    # Transcribing the audio to text
    transcript = aai.Transcriber().transcribe(filepath, cfg)
    
    # Calculating average confidence score for sentiment analysis
    sum_confidence = 0
    count = 0
    for sentiment_result in transcript.sentiment_analysis:
        sum_confidence += sentiment_result.confidence
        count += 1
    avg_confidence = sum_confidence / count
    
    # Display sentiment analysis result
    if 0 < avg_confidence <= 0.5:
        st.info("Not Satisfied.......😒😒😒")
    else:
        st.info("Satisfied.......😀😀😀") 
    st.info(avg_confidence)
    
    # Plotting sentiment analysis result
    labels = []
    scores = []
    for sentiment_result in transcript.sentiment_analysis:
        labels.append(sentiment_result.sentiment)
        scores.append(sentiment_result.confidence)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(labels, scores)
    ax.set_xticks(labels)
    ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(fig)
    
    # Highlighting important words in the audio
    cfig = aai.TranscriptionConfig(auto_highlights=True)
    transcript = aai.Transcriber().transcribe(filepath, config=cfig)
    
    # Displaying the highlighted words and their ranks
    for result in transcript.auto_highlights.results:
        st.info(f"Highlight: {result.text},  Rank: {result.rank}")

def main():
    st.title("Customer Satisfaction Analysis")

    uploaded_file = st.file_uploader("Please upload audio file", type=["mp3", "wav"])
    if uploaded_file is not None:
        if st.button("Transcript"):
            with st.spinner("Processing....😊"):
                audio_processing(uploaded_file.name)

if __name__ == "__main__":
    main()
