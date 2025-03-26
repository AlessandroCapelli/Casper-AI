import { CommonModule } from '@angular/common';
import { Component, signal, ViewChild, ElementRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatMessage, Service } from './app.service';

@Component({
	selector: 'app-root',
	standalone: true,
	imports: [CommonModule, FormsModule],
	templateUrl: './app.component.html',
	styleUrl: './app.component.css'
})
export class AppComponent {
	@ViewChild('scrollMe') private scrollContainer!: ElementRef;

	protected readonly chatHistory = signal<ChatMessage[]>([]);
	protected readonly userPrompt = signal<ChatMessage>({ sender: 'user', text: '', timestamp: new Date() });
	protected readonly isLoading = signal(false);

	constructor(private readonly service: Service) {
		this.chatHistory.set(JSON.parse(localStorage.getItem('chatHistory') ?? '[]'));
	}

	protected async sendMessageAsync(): Promise<void> {
		if (!this.userPrompt().text.trim())
			return;

		this.isLoading.set(true);
		this.userPrompt.update(prompt => ({ ...prompt, timestamp: new Date() }));
		this.chatHistory.update(history => [...history, this.userPrompt()]);

		try {
			const response = await this.service.sendMessageAsync(this.userPrompt().text.trim(), this.chatHistory());
			this.chatHistory.update(history => [...history, {
				sender: 'ai',
				text: response.reply,
				timestamp: new Date()
			}]);
		} catch (error) {
			this.chatHistory.update(history => [...history, {
				sender: 'error',
				text: (error as Error).message,
				timestamp: new Date()
			}]);
		} finally {
			localStorage.setItem('chatHistory', JSON.stringify(this.chatHistory()));
			this.userPrompt.set({ sender: 'user', text: '', timestamp: new Date() });
			this.isLoading.set(false);

			this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
		}
	}
}
