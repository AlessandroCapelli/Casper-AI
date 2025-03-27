export const environment = {
	production: false,
	get apiUrl(): string {
		const protocol = window.location.protocol;
		const hostname = window.location.hostname;

		return `${protocol}//${hostname}:5000`;
	}
};
