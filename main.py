import requests
import openai
from pydub import AudioSegment
import os
import uuid
import tiktoken

token_limit = 4000

# Break the video up openAI can only take in 25 mb lets use 20 mb to buy us some time
def breakVidUp(input_file_path):
    input_file = AudioSegment.from_mp3(input_file_path)
    # Get size in MB
    file_size = os.path.getsize(input_file_path) / 1048576

    # Number of chunks we'll have to break the video up in
    num_chunks = int(file_size / 20) + 1

    # Get the number of miliseconds per chunk
    num_miliseconds_per_chunk = (input_file.duration_seconds * 1000) / num_chunks

    chunks = []

    for i in range(num_chunks):
        chunk = input_file[(i * num_miliseconds_per_chunk) : (i * num_miliseconds_per_chunk) + num_miliseconds_per_chunk]
        chunks.append(chunk)

    return chunks


# This piece will get a transcript a chunk 
def getTranscript(chunk):
    # Create a temp file to store the chunk in and transcribe it.
    temp_file_path = uuid.uuid4().hex + '.mp3'

    chunk.export(temp_file_path, format='mp3')

    audio_file = open(temp_file_path, 'rb')

    transcript = openai.Audio.transcribe('whisper-1', audio_file)

    # clean up and remove the file
    os.remove(temp_file_path)

    return transcript

# Gets the number of tokens in text
def getNumTokens(text, encoding):
    input_ids = encoding.encode(text)
    num_tokens = len(input_ids)
    return num_tokens

# Breaks up the text into chunk size
def breakTextIntoChunks(text, encoding, chunk_size=2000, overlap=100):
    tokens = encoding.encode(text)

    num_tokens = len(tokens)

    chunks = []

    for i in range(0, num_tokens, chunk_size - overlap):
        chunk = tokens[i: i + chunk_size]
        chunks.append(chunk)

    return chunks

# Break the transcript up based on the tokens
def breakTranscriptUp(encoding, transcript):
    input_ids = encoding.encode(transcript)
    num_tokens = len(input_ids)

    stringSize = token_limit / 2
    if num_tokens > stringSize:
        res = []
        chunks = breakTextIntoChunks(transcript, encoding)

        for chunk in chunks:
            res.append(encoding.decode(chunk))

        return res

    return [transcript]

# Now pop that stuff into gpt
def getNotes(transcript):
    response = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages=[
            {"role": "user", "content": transcript + "- This is a snippet from a lecture. Could you take bulleted notes on this lecture as a college student would?" }
        ]
    )

    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    input_file_path = input('Enter the file path of the audio you want to take notes on: ')

    output_file = input('Enter the output file: ')

    vid_chunks = breakVidUp(input_file_path)
    notes = []

    for vid_chunk in vid_chunks:
        transcription = getTranscript(vid_chunk)

        # break each transcritpion up
        broken_transcriptions = breakTranscriptUp(encoding, transcription['text'])

        for broken_transcription in broken_transcriptions:
            chunk_notes = getNotes(broken_transcription)
            notes.append(chunk_notes)

    f = open(output_file, "w")

    for note in notes:
        f.write(note)
        f.write('\n')
