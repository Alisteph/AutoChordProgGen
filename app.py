import streamlit as st
import chord_gen


st.title('AutoChordProgressGenerater')


init_chords = st.text_input("input")
exec_generate = st.button('exec')

if exec_generate:
    output_chord = chord_gen.generate_chord(init_chords.split())

    output = st.text(''.join(output_chord))
    button_downloado = st.download_button(
        'download onchord',
        open('tmp/chord6o.mid', 'br'),
        'chord6o.mid'
    )
    button_downloadn = st.download_button(
        'download norm',
        open('tmp/chord6n.mid', 'br'),
        'chord6n.mid'
    )
