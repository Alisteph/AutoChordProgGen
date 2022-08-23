import streamlit as st
import chord_gen
import display_txts as dtx


st.title('AutoChordProgressGenerater')


init_chords = st.text_input("initial chords")
num_chords = st.text_input("number of output chords")
exec_generate = st.button('exec')


if exec_generate:
    if len(init_chords.split()) == 0:
        st.markdown(dtx.no_ini_chord, unsafe_allow_html=True)
    elif len(num_chords.split()) == 0 or len(num_chords.split()) > 1:
        st.markdown(dtx.not1num, unsafe_allow_html=True)
    else:
        output_chord = chord_gen.generate_chord(init_chords.split())
        if len(output_chord) > int(num_chords):
            output_chord = output_chord[:int(num_chords)]

        st.code(' '.join(output_chord))

    button_downloado = st.download_button(
        'download onchord',
        open('on_chord.mid', 'br'),
        'on_chord.mid'
    )
    button_downloadn = st.download_button(
        'download normal chord',
        open('chord.mid', 'br'),
        'chord.mid'
    )

st.markdown(dtx.description1)