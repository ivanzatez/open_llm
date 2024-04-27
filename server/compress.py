from llmlingua import PromptCompressor


text = """
A few words about Dostoevsky himself may help the English reader to understand his work.

Dostoevsky was the son of a doctor. His parents were very hard-working and deeply religious people, but so poor that they lived with their five children in only two rooms. The father and mother spent their evenings in reading aloud to their children, generally from books of a serious character.

Though always sickly and delicate Dostoevsky came out third in the final examination of the Petersburg school of Engineering. There he had already begun his first work, “Poor Folk.”

This story was published by the poet Nekrassov in his review and was received with acclamations. The shy, unknown youth found himself instantly something of a celebrity. A brilliant and successful career seemed to open before him, but those hopes were soon dashed. In 1849 he was arrested.

Though neither by temperament nor conviction a revolutionist, Dostoevsky was one of a little group of young men who met together to read Fourier and Proudhon. He was accused of “taking part in conversations against the censorship, of reading a letter from Byelinsky to Gogol, and of knowing of the intention to set up a printing press.” Under Nicholas I. (that “stern and just man,” as Maurice Baring calls him) this was enough, and he was condemned to death. After eight months’ imprisonment he was with twenty-one others taken out to the Semyonovsky Square to be shot. Writing to his brother Mihail, Dostoevsky says: “They snapped words over our heads, and they made us put on the white shirts worn by persons condemned to death. Thereupon we were bound in threes to stakes, to suffer execution. Being the third in the row, I concluded I had only a few minutes of life before me. I thought of you and your dear ones and I contrived to kiss Plestcheiev and Dourov, who were next to me, and to bid them farewell. Suddenly the troops beat a tattoo, we were unbound, brought back upon the scaffold, and informed that his Majesty had spared us our lives.” The sentence was commuted to hard labour.

One of the prisoners, Grigoryev, went mad as soon as he was untied, and never regained his sanity.
Russian critic, who seeks to explain the feeling inspired by Dostoevsky: “He was one of ourselves, a man of our blood and our bone, but one who has suffered and has seen so much more deeply than we have his insight impresses us as wisdom... that wisdom
"""

instruction = "You are a robot. Your job is to listen to the user"




# Init llm lingua2 prompt compression
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu" # Whether to use llmlingua-2
)




compressed_prompt = llm_lingua.compress_prompt(
    context=[text],
    instruction=instruction,
    question= "Can you please summarize the text for me?"
)
print(compressed_prompt)