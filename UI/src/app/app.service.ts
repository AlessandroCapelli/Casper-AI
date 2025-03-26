import { Injectable } from '@angular/core';

export interface ChatMessage {
	sender: 'user' | 'ai' | 'error';
	text: string;
	timestamp: Date;
}

export interface ChatResponse {
	reply: string;
}

const API_URL = 'http://localhost:5000/chat';

@Injectable({ providedIn: 'root' })
export class Service {
	public async sendMessageAsync(userMessage: string, history: ChatMessage[]): Promise<ChatResponse> {
		try {
			const response = await fetch(API_URL, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ message: userMessage, history: history })
			});

			if (!response.ok)
				throw new Error(response.statusText);

			return await response.json();
		} catch {
			throw new Error('Failed to send message. Please try again later.');
		}
	}
}
