import { Injectable } from '@angular/core';
import { environment } from '../../environment';

export interface ChatMessage {
	sender: 'User' | 'Assistant' | 'Error';
	text: string;
	timestamp: Date;
}

export interface ChatResponse {
	reply: string;
}

const API_URL = `${environment.apiUrl}/chat`;

@Injectable({ providedIn: 'root' })
export class Service {
	public async sendMessageAsync(history: ChatMessage[]): Promise<ChatResponse> {
		try {
			const response = await fetch(API_URL, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ history: history })
			});

			if (!response.ok)
				throw new Error(response.statusText);

			return await response.json();
		} catch {
			throw new Error('Failed to send message. Please try again later.');
		}
	}
}
