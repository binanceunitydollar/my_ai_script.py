import torch
import torchaudio

# Load the pre-trained voice recognition model
model = torch.hub.load('pytorch/audio', 'google_speech_command_classification_v0')

# Set the model to evaluation mode
model.eval()

# Function to perform voice recognition
def recognize_voice(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)

    # Preprocess the audio waveform
    transformed = model.transform(waveform)
    transformed = transformed.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(transformed)

    # Get the predicted class label
    predicted_index = output.argmax(1).item()
    labels = ['unknown', 'silence', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off']
    predicted_label = labels[predicted_index]

    return predicted_label

# Main program loop
if __name__ == '__main__':
    while True:
        audio_file = input("Enter the path to the audio file: ")
        if audio_file == 'exit':
            break

        predicted_label = recognize_voice(audio_file)
        print("Predicted label:", predicted_label)
