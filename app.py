import streamlit as st
import chord_gen


st.title('AutoChordProgressGenerater')


init_chords = st.text_input("input")
num_chords = st.text_input("number of chords")
exec_generate = st.button('exec')

if exec_generate:
    output_chord = chord_gen.generate_chord(init_chords.split())
    if len(output_chord) > int(num_chords):
        output_chord = output_chord[:int(num_chords)]

    output = st.text(' '.join(output_chord))

    st.write('MIDI download function is in progress...')
    # button_downloado = st.download_button(
    #     'download onchord',
    #     open('tmp/on_chord.mid', 'br'),
    #     'on_chord.mid'
    # )
    # button_downloadn = st.download_button(
    #     'download normal chord',
    #     open('tmp/chord.mid', 'br'),
    #     'chord.mid'
    # )
