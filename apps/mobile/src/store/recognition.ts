import { create } from 'zustand';
import type { RecognitionEvent } from '../types/recognition';

type State = {
  current?: RecognitionEvent;
  events: RecognitionEvent[];
  setEvent: (value: RecognitionEvent) => void;
};

export const useRecognitionStore = create<State>((set) => ({
  events: [],
  setEvent: (current) => {
    set((state) => {
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
