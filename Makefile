all:
	cmake -S ./src/ -B ./build
	make -C build

clean:
	rm -rf build/*