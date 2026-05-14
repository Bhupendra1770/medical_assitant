"""
Medical Response Generator
Uses Groq LLM with RAG context from pgvector knowledge base
"""

import logging
import re
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

MEDICAL_SYSTEM_PROMPT = """You are MediAssist, an intelligent medical information assistant.

Your role:
- Help users understand symptoms, conditions, and general medication information
- Provide clear, empathetic, and accurate medical information
- Always recommend consulting a licensed doctor for diagnosis or treatment
- Use any provided medical reference context to give evidence-based answers
- Keep voice responses concise (2-4 sentences) and text responses well-structured

When a user describes symptoms:
1. Acknowledge their concern empathetically
2. Explain possible causes based on the symptoms
3. Suggest common remedies or OTC medications if appropriate
4. Always recommend seeing a doctor for proper diagnosis

When asked about medications:
- Explain usage, typical dosage ranges, and common side effects
- Mention contraindications if relevant
- Remind that prescriptions require a licensed physician

⚠️ IMPORTANT DISCLAIMER: Always end your response with a brief reminder that this is for informational purposes only and not a substitute for professional medical advice.

Respond in clear, simple language that a patient can understand. Be warm and reassuring.
"""


class MedicalResponseGenerator:
    def __init__(self, groq_api_key: str, rag_service=None):
        self.rag_service = rag_service
        self.conversation_history: List[Dict] = []
        self.groq_client = None
        self._init_groq(groq_api_key)

    def _init_groq(self, api_key: str):
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=api_key)
            logger.info("✅ Groq client initialized")
        except ImportError:
            logger.error("❌ Groq not installed")

    async def generate(self, user_text: str, is_voice: bool = False) -> str:
        """
        Generate a medical response with RAG context.
        is_voice=True produces shorter, TTS-friendly responses.
        """
        try:
            # 1. Retrieve RAG context
            rag_context = ""
            if self.rag_service:
                rag_context = self.rag_service.get_context(user_text)

            # 2. Build messages
            messages = self._build_messages(user_text, rag_context, is_voice)

            # 3. Call LLM
            response = await self._call_groq(messages, is_voice)

            # 4. Store history
            self._push_history(user_text, response)

            return response

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I apologize, I'm having trouble responding right now. Please try again or consult a healthcare professional."

    def _build_messages(self, user_text: str, rag_context: str, is_voice: bool) -> List[Dict]:
        """Compose the full message list for the LLM"""
        # Tailor system prompt for voice (shorter output)
        sys_prompt = MEDICAL_SYSTEM_PROMPT
        if is_voice:
            sys_prompt += "\n\nIMPORTANT: This is a voice response. Keep it under 3 sentences. Be warm and direct."
        else:
            sys_prompt += "\n\nThis is a text/chat response. You may use markdown formatting, bullet points, and be more detailed."

        # Inject RAG context if available
        if rag_context:
            sys_prompt += f"\n\n--- MEDICAL REFERENCE CONTEXT ---\n{rag_context}\n--- END CONTEXT ---"

        messages = [{"role": "system", "content": sys_prompt}]

        # Add recent conversation history (last 3 turns)
        for turn in self.conversation_history[-3:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": user_text})
        return messages

    async def _call_groq(self, messages: List[Dict], is_voice: bool) -> str:
        if not self.groq_client:
            return "Sorry, the AI service is not available right now."

        max_tokens = 300 if is_voice else 1024

        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.5,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
            return self._clean_response(text, is_voice)
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return "I'm having trouble reaching the AI service. Please try again."

    def _clean_response(self, text: str, is_voice: bool) -> str:
        """Remove markdown artifacts that don't work well in voice"""
        if not is_voice:
            return text

        # Strip markdown for TTS
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)       # italic
        text = re.sub(r"#{1,6}\s", "", text)            # headers
        text = re.sub(r"- ", "", text)                  # bullet points
        text = re.sub(r"\n+", " ", text)                # newlines
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _push_history(self, user_text: str, response: str):
        self.conversation_history.append({"user": user_text, "assistant": response})
        # Keep last 6 turns
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]

    def clear_history(self):
        self.conversation_history = []


# ── Global singleton ────────────────────────────────────────────────────────

_generator: Optional[MedicalResponseGenerator] = None


def get_generator() -> Optional[MedicalResponseGenerator]:
    return _generator


def initialize_generator(groq_api_key: str, rag_service=None) -> MedicalResponseGenerator:
    global _generator
    _generator = MedicalResponseGenerator(groq_api_key, rag_service)
    return _generator
