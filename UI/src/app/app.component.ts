import { CommonModule } from '@angular/common';
import { Component, signal, ViewChild, ElementRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatMessage, Service } from './app.service';
import { Pipe, PipeTransform } from '@angular/core';
import { forwardRef } from "@angular/core";
import { marked } from 'marked';

@Component({
	selector: 'app-root',
	standalone: true,
	imports: [CommonModule, FormsModule, forwardRef(() => MarkdownPipe)],
	templateUrl: './app.component.html',
	styleUrl: './app.component.css'
})
export class AppComponent {
	@ViewChild('scrollMe') private scrollContainer!: ElementRef;

	protected readonly chatHistory = signal<ChatMessage[]>([]);
	protected readonly userPrompt = signal<ChatMessage>({ sender: 'User', text: '', timestamp: new Date() });
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
			const response = await this.service.sendMessageAsync(this.chatHistory());
			this.chatHistory.update(history => [...history, {
				sender: 'Assistant',
				text: response.reply,
				timestamp: new Date()
			}]);
		} catch (error) {
			this.chatHistory.update(history => [...history, {
				sender: 'Error',
				text: (error as Error).message,
				timestamp: new Date()
			}]);
		} finally {
			localStorage.setItem('chatHistory', JSON.stringify(this.chatHistory()));
			this.userPrompt.set({ sender: 'User', text: '', timestamp: new Date() });
			this.isLoading.set(false);

			this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
		}
	}

	protected clearChat(): void {
		this.chatHistory.set([]);
		localStorage.removeItem('chatHistory');
		this.userPrompt.set({ sender: 'User', text: '', timestamp: new Date() });
	}
}

@Pipe({
	name: 'markdown'
})
export class MarkdownPipe implements PipeTransform {
	public transform(value: string): string {
		marked.setOptions({
			gfm: true,
			breaks: true,
		});

		return marked(value) as string;
	}
}