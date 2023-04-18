class Tutor:
    """
    Tutor is a class that uses langchain to create a chatbot that acts as a
    tutor.

    Args:

    student_profile: StudentProfile
        The student profile contains information about the student that
        the tutor will use to personalize the tutoring experience.

    Methods:

    setup_chatbot: None -> PromptTemplate
        This method sets up the chatbot by creating a prompt template that
        the chatbot will use to generate responses.

    setup_chain: None -> ConversationChain
        This method sets up the chatbot chain by creating a chain that
        contains the chatbot and the prompt template.

    get_history: None -> str
        This method returns the chatbot's history as a string.

    say: str -> str
        This method takes in a string and returns a string that is the
        chatbot's response to the input string.
    """

    def __init__(self, student: StudentProfile):
        self.student_profile = student
        self.setup_chain()

    def setup_chatbot(self) -> PromptTemplate:
        template_base = f"""
        Teach.me is a learning assistant that's goal is to help anyone,
        learn anything.  To do this, it learns to adapt to the student's
        needs and learning style.  Teach.me will guide the student in their
        learning journey by being helpful, patient, and encouraging.
        Teach.me tries to identify what core concepts the student is
        struggling with and then tries to help the student understand them
        better.  Teach.me will try not to give the student the answer, but
        help them find it themselves.  If the student is stuck, Teach.me
        will try to help them get unstuck by giving hints and suggestions.
        If Teach.me has to give the student the answer, it will ask another
        question targeting the same concept before moving on.  Please always
        speak to the student in a way that would be appropriate according to
        their grade level.

        Student Info:
        Name: {self.student_profile.name}
        Grade: {self.student_profile.grade}
        Subject: {self.student_profile.subject}
        Interests: {self.student_profile.interests}

        Teach.me: Hello, I'm Teach.me.  What would you like to learn about today?
        """
        templae_format = """{history}
        Student: {human_input}
        Teach.me:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=template_base + templae_format,
        )
        return prompt

    def setup_chain(self):
        prompt = self.setup_chatbot()

        chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0),
            prompt=prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=2),
        )
        self.chain = chatgpt_chain

    def get_history(self) -> str:
        return self.chain.memory.get_history()

    def say(self, text: str) -> str:
        return self.chain.predict(human_input=text)
