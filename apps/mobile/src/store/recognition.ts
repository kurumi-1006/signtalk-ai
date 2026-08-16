import { create } from 'zustand';
import { type RecognitionEvent, recognitionEventSchema } from '@signtalk/contracts';

type State = {
  current?: RecognitionEvent;
  events: RecognitionEvent[];
  setEvent: (value: unknown) => void;
};

export const useRecognitionStore = create<State>((set) => ({
  events: [],
  setEvent: (value) => {
    const result = recognitionEventSchema.safeParse(value);
    if (!result.success) return;
    set((state) => {
      const current = result.data;
      if (current.payload.accepted === false) {
        return {
          current,
          events: state.events.filter((event) => event.payload.accepted !== false),
        };
      }
      return {
        current,
        events: [
          current,
          ...state.events.filter((event) => event.eventId !== current.eventId),
        ].slice(0, 20),
      };
    });
  },
}));
