import argparse
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import *


def build_chatbot(hourly: pd.DataFrame, M: dict):
    series = hourly["kwh"]
    start = series.index[-1] + pd.Timedelta(hours=1)
    history = prepare_history(series, start)
    next_day = forecast_next_day(hourly, M)
    best_name = M["best_name"]
    best_mae, _, _, best_wape = M["results"][best_name]

    peak_ts = next_day.idxmax()
    peak_hour, peak_load = int(peak_ts.hour), float(next_day.max())
    offpeak = next_day[[t.hour not in PEAK_HOURS for t in next_day.index]]
    cheap_hour = int(offpeak.idxmin().hour)
    daily_kwh = float(next_day.sum())
    est_cost = sum(v * (TARIFF_PEAK if t.hour in PEAK_HOURS else TARIFF_OFFPEAK) for t, v in next_day.items())
    peak_share = float(next_day[[t.hour in PEAK_HOURS for t in next_day.index]].sum() / daily_kwh * 100)
    typical_daily = float(series.resample("D").sum(min_count=20).mean())
    vs_typical = (daily_kwh / typical_daily - 1) * 100
    appliance_share = hourly[["kitchen", "laundry", "water_heater_ac"]].sum()
    top_key = appliance_share.idxmax()
    top_pct = appliance_share[top_key] / appliance_share.sum() * 100
    top_app = {"kitchen": "kitchen appliances", "laundry": "laundry appliances",
               "water_heater_ac": "water heater / air conditioner"}[top_key]
    span = f"{next_day.index[0]:%d %b %H:%M} to {next_day.index[-1]:%d %b %H:%M}"

    intents = {
        "forecast": ("how much electricity will i use tomorrow forecast prediction usage cost bill next 24 hours "
                     "consumption expected total",
                     lambda: f"Forecast for the next 24 hours ({span}): about {daily_kwh:.1f} kWh, "
                             f"estimated cost R{est_cost:.0f}."),
        "peak": ("when is my peak time highest demand busiest hour peak load peak demand",
                 lambda: f"Your peak load is expected around {peak_hour:02d}:00 "
                         f"at about {peak_load:.1f} kW (about {peak_load:.2f} kWh in that hour)."),
        "compare": ("is tomorrow higher lower than usual normal average compare",
                    lambda: f"The next 24 hours are forecast to be {abs(vs_typical):.0f}% "
                            f"{'above' if vs_typical >= 0 else 'below'} your typical day ({typical_daily:.1f} kWh)."),
        "save": ("how can i save money reduce my usage bill cheaper cut costs lower peak load shift "
                 "consumption away manage demand",
                 lambda: f"Move heavy appliances (washing, geyser, dishwasher) to about "
                         f"{cheap_hour:02d}:00 and avoid peak times (07:00-10:00 and 18:00-21:00), "
                         f"when {peak_share:.0f}% of your forecast usage falls. "
                         f"Your biggest metered consumer is the {top_app}."),
        "appliance": ("which appliance uses the most electricity biggest consumer",
                      lambda: f"Most of your sub-metered energy ({top_pct:.0f}%) goes to the {top_app}."),
        "accuracy": ("how accurate is the model forecast error reliable trust",
                     lambda: f"The {best_name} model is off by about {best_mae:.2f} kWh per hour on average, "
                             f"about {best_wape:.0f}% of your total load, on unseen test days."),
        "outage": ("load shedding blackout outage power cut grid overload",
                   lambda: "Lower demand during peak hours helps the grid avoid overload and outages."),
        "greet": ("hello hi hey good morning",
                  lambda: "Hello! Ask me about tomorrow's usage, peak load, accuracy or ways to save. "
                          "Type 'help' to see example questions."),
    }
    help_questions = [
        "How much power will I use in the next hour?",
        "What is my forecast for the next 3 hours?",
        "How much electricity will I use in the next 24 hours?",
        "What will it cost me?",
        "When is my peak load? / Which hour has the highest demand?",
        "Is tomorrow higher than usual?",
        "How can I save money? / How can I lower my peak load?",
        "Which appliance uses the most electricity?",
        "How accurate is the forecast?",
        "What is load shedding / how does demand affect outages?",
    ]
    help_text = ("You can ask me things like:\n"
                 + "\n".join(f"  {i}. {q}" for i, q in enumerate(help_questions, 1))
                 + "\nType 'quit' to exit.")
    names = list(intents)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform([intents[k][0] for k in names])
    last = series.index[-1]

    def short_horizon(question: str):
        q = question.lower()
        asks = re.search(r"\b(forecast|predict|use|usage|consum\w*|expect\w*|how much)\b", q)
        if not asks or "peak" in q or "highest" in q:
            return None
        m = re.search(r"(\d+)\s*(?:hours?|hrs?)\b", q)
        n = int(m.group(1)) if m else (1 if re.search(r"\bnext hour\b", q) else None)
        if n is None or not 1 <= n <= 23:
            return None
        steps = forecast_ml(M["hour_final"], FEATURES_SHORT, history, start, n)
        if n == 1:
            return (f"Forecast for the next hour ({steps.index[0]:%H:00}): about {steps.iloc[0]:.2f} kWh "
                    f"(after the last record at {last:%Y-%m-%d %H:%M}).")
        detail = ", ".join(f"{t:%H:00} {v:.2f} kWh" for t, v in steps.items())
        return (f"Forecast for the next {n} hours: {detail}. Total about {steps.sum():.2f} kWh "
                f"(after the last record at {last:%Y-%m-%d %H:%M}).")

    def chatbot(question: str) -> str:
        if question.strip().lower().strip("?!. ") in {"help", "menu", "options", "what can you do"}:
            return help_text
        quick = short_horizon(question)
        if quick:
            return quick
        sims = cosine_similarity(vec.transform([question.lower()]), matrix)[0]
        if sims.max() < 0.05:
            return "Sorry, I did not understand that. Type 'help' to see what you can ask."
        return intents[names[sims.argmax()]][1]()

    return chatbot, next_day


def run_chat(chatbot, voice: bool = False) -> None:
    if voice:
        try:
            import pyttsx3
            import speech_recognition as sr
        except ImportError:
            raise SystemExit("Voice mode needs: pip install SpeechRecognition pyaudio pyttsx3")
        recognizer, engine = sr.Recognizer(), pyttsx3.init()
        print("\nVoice chatbot ready. Say 'quit' to exit.")
        while True:
            with sr.Microphone() as src:
                print("Listening...")
                audio = recognizer.listen(src)
            try:
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                print("Sorry, I could not hear that.")
                continue
            if text.strip().lower() in {"quit", "exit"}:
                break
            reply = chatbot(text)
            print("You:", text, "\nBot:", reply)
            engine.say(reply)
            engine.runAndWait()
        return
    print("\nChatbot ready. Type 'help' to see what you can ask, or 'quit' to exit.")
    while True:
        try:
            q = input("You: ")
        except EOFError:
            break
        if q.strip().lower() in {"quit", "exit"}:
            break
        print("Bot:", chatbot(q))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Household energy advice chatbot")
    parser.add_argument("--voice", action="store_true", help="use microphone and speaker")
    args = parser.parse_args()
    hourly = clean_data()
    chatbot, _ = build_chatbot(hourly, require_models())
    run_chat(chatbot, voice=args.voice)
