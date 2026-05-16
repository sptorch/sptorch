// SPTorch Tang9k UART responder bring-up design.
//
// This bitstream is intentionally tiny: it validates one SPTorch serial-v1
// Ping frame from the host and replies with a Pong frame that echoes the same
// sequence number.  It is the first real-board smoke test for the HAL serial
// contract before any MatMul/DMA logic is introduced.

module tang9k_uart_responder (
    input  wire       clk,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [5:0] led
);
    localparam integer CLK_HZ = 27000000;
    localparam integer BAUD = 115200;
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam integer HALF_CLKS_PER_BIT = CLKS_PER_BIT / 2;

    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME  = 32'h01000193;

    localparam [7:0] SERIAL_MAGIC_S = 8'h53;
    localparam [7:0] SERIAL_MAGIC_P = 8'h50;
    localparam [7:0] SERIAL_VERSION = 8'h01;
    localparam [7:0] OPCODE_PING   = 8'h01;
    localparam [7:0] OPCODE_PONG   = 8'h02;
    localparam [7:0] HEADER_LEN     = 8'd16;
    localparam [7:0] CHECKSUM_LEN   = 8'd4;
    localparam [15:0] MAX_PAYLOAD_LEN = 16'd64;

    localparam [3:0] RX_WAIT_S  = 4'd0;
    localparam [3:0] RX_WAIT_P  = 4'd1;
    localparam [3:0] RX_HEADER  = 4'd2;
    localparam [3:0] RX_PAYLOAD = 4'd3;
    localparam [3:0] RX_CSUM    = 4'd4;
    localparam [3:0] RX_PAD     = 4'd5;

    localparam [1:0] TX_SEND      = 2'd0;
    localparam [1:0] TX_WAIT_BUSY = 2'd1;
    localparam [1:0] TX_WAIT_IDLE = 2'd2;

    wire       rx_valid;
    wire [7:0] rx_byte;

    reg        tx_start;
    reg  [7:0] tx_data;
    wire       tx_busy;

    uart_rx_8n1 #(
        .CLKS_PER_BIT(CLKS_PER_BIT),
        .HALF_CLKS_PER_BIT(HALF_CLKS_PER_BIT)
    ) uart_rx_i (
        .clk(clk),
        .rx(uart_rx),
        .data(rx_byte),
        .valid(rx_valid)
    );

    uart_tx_8n1 #(
        .CLKS_PER_BIT(CLKS_PER_BIT)
    ) uart_tx_i (
        .clk(clk),
        .start(tx_start),
        .data(tx_data),
        .tx(uart_tx),
        .busy(tx_busy)
    );

    reg [3:0]  rx_state = RX_WAIT_S;
    reg [4:0]  header_index = 5'd0;
    reg [15:0] payload_len = 16'd0;
    reg [15:0] payload_count = 16'd0;
    reg [1:0]  checksum_index = 2'd0;
    reg [31:0] checksum_seen = 32'd0;
    reg [31:0] checksum_calc = FNV_OFFSET;
    reg [2:0]  pad_remaining = 3'd0;
    reg [31:0] sequence = 32'd0;

    reg        tx_frame_active = 1'b0;
    reg [4:0]  tx_frame_index = 5'd0;
    reg [1:0]  tx_phase = TX_SEND;
    reg [31:0] tx_sequence = 32'd0;
    reg [31:0] tx_checksum = 32'd0;
    reg        pong_request_toggle = 1'b0;
    reg        pong_request_seen = 1'b0;
    reg [31:0] pong_request_sequence = 32'd0;

    reg [25:0] heartbeat = 26'd0;
    reg        led_heartbeat = 1'b0;
    reg        led_rx_byte = 1'b0;
    reg        led_tx_byte = 1'b0;
    reg        led_protocol_error = 1'b0;
    reg        led_ping_accept = 1'b0;
    reg        led_checksum_error = 1'b0;

    assign led = ~{
        led_checksum_error,
        led_ping_accept,
        led_protocol_error,
        led_tx_byte,
        led_rx_byte,
        led_heartbeat
    };

    function [31:0] fnv_feed;
        input [31:0] hash;
        input [7:0]  byte_value;
        begin
            fnv_feed = (hash ^ {24'd0, byte_value}) * FNV_PRIME;
        end
    endfunction

    function [2:0] frame_padding_len;
        input [15:0] payload_length;
        reg [15:0] total_len;
        reg [2:0] rem;
        begin
            total_len = payload_length + HEADER_LEN + CHECKSUM_LEN;
            rem = total_len[2:0];
            frame_padding_len = rem == 3'd0 ? 3'd0 : (3'd0 - rem);
        end
    endfunction

    function [31:0] pong_checksum;
        input [31:0] seq;
        reg [31:0] h;
        begin
            h = FNV_OFFSET;
            h = fnv_feed(h, SERIAL_MAGIC_S);
            h = fnv_feed(h, SERIAL_MAGIC_P);
            h = fnv_feed(h, SERIAL_VERSION);
            h = fnv_feed(h, OPCODE_PONG);
            h = fnv_feed(h, seq[7:0]);
            h = fnv_feed(h, seq[15:8]);
            h = fnv_feed(h, seq[23:16]);
            h = fnv_feed(h, seq[31:24]);
            h = fnv_feed(h, 8'h00);
            h = fnv_feed(h, 8'h00);
            h = fnv_feed(h, 8'h00);
            h = fnv_feed(h, 8'h00);
            h = fnv_feed(h, 8'h00);
            h = fnv_feed(h, 8'h00);
            h = fnv_feed(h, 8'h00);
            h = fnv_feed(h, 8'h00);
            pong_checksum = h;
        end
    endfunction

    function [7:0] pong_byte;
        input [4:0] index;
        input [31:0] seq;
        input [31:0] csum;
        begin
            case (index)
                5'd0:  pong_byte = SERIAL_MAGIC_S;
                5'd1:  pong_byte = SERIAL_MAGIC_P;
                5'd2:  pong_byte = SERIAL_VERSION;
                5'd3:  pong_byte = OPCODE_PONG;
                5'd4:  pong_byte = seq[7:0];
                5'd5:  pong_byte = seq[15:8];
                5'd6:  pong_byte = seq[23:16];
                5'd7:  pong_byte = seq[31:24];
                5'd16: pong_byte = csum[7:0];
                5'd17: pong_byte = csum[15:8];
                5'd18: pong_byte = csum[23:16];
                5'd19: pong_byte = csum[31:24];
                default: pong_byte = 8'h00;
            endcase
        end
    endfunction

    task reset_parser;
        begin
            rx_state <= RX_WAIT_S;
            header_index <= 5'd0;
            payload_len <= 16'd0;
            payload_count <= 16'd0;
            checksum_index <= 2'd0;
            checksum_seen <= 32'd0;
            checksum_calc <= FNV_OFFSET;
            pad_remaining <= 3'd0;
        end
    endtask

    task accept_ping;
        begin
            pong_request_sequence <= sequence;
            pong_request_toggle <= ~pong_request_toggle;
            led_ping_accept <= ~led_ping_accept;
            reset_parser();
        end
    endtask

    always @(posedge clk) begin
        heartbeat <= heartbeat + 26'd1;
        if (heartbeat == 26'd0) begin
            led_heartbeat <= ~led_heartbeat;
        end
    end

    always @(posedge clk) begin
        tx_start <= 1'b0;

        if (!tx_frame_active && (pong_request_seen != pong_request_toggle)) begin
            pong_request_seen <= pong_request_toggle;
            tx_sequence <= pong_request_sequence;
            tx_checksum <= pong_checksum(pong_request_sequence);
            tx_frame_index <= 5'd0;
            tx_phase <= TX_SEND;
            tx_frame_active <= 1'b1;
        end else if (tx_frame_active) begin
            case (tx_phase)
                TX_SEND: begin
                    if (!tx_busy) begin
                        tx_data <= pong_byte(tx_frame_index, tx_sequence, tx_checksum);
                        tx_start <= 1'b1;
                        led_tx_byte <= ~led_tx_byte;
                        tx_phase <= TX_WAIT_BUSY;
                    end
                end

                TX_WAIT_BUSY: begin
                    if (tx_busy) begin
                        tx_phase <= TX_WAIT_IDLE;
                    end
                end

                TX_WAIT_IDLE: begin
                    if (!tx_busy) begin
                        if (tx_frame_index == 5'd23) begin
                            tx_frame_active <= 1'b0;
                            tx_frame_index <= 5'd0;
                            tx_phase <= TX_SEND;
                        end else begin
                            tx_frame_index <= tx_frame_index + 5'd1;
                            tx_phase <= TX_SEND;
                        end
                    end
                end

                default: tx_phase <= TX_SEND;
            endcase
        end
    end

    always @(posedge clk) begin
        if (rx_valid) begin
            led_rx_byte <= ~led_rx_byte;

            case (rx_state)
                RX_WAIT_S: begin
                    if (rx_byte == SERIAL_MAGIC_S) begin
                        rx_state <= RX_WAIT_P;
                    end
                end

                RX_WAIT_P: begin
                    if (rx_byte == SERIAL_MAGIC_P) begin
                        checksum_calc <= fnv_feed(fnv_feed(FNV_OFFSET, SERIAL_MAGIC_S), SERIAL_MAGIC_P);
                        header_index <= 5'd2;
                        payload_len <= 16'd0;
                        payload_count <= 16'd0;
                        rx_state <= RX_HEADER;
                    end else if (rx_byte != SERIAL_MAGIC_S) begin
                        rx_state <= RX_WAIT_S;
                    end
                end

                RX_HEADER: begin
                    checksum_calc <= fnv_feed(checksum_calc, rx_byte);
                    case (header_index)
                        5'd2: begin
                            if (rx_byte != SERIAL_VERSION) begin
                                led_protocol_error <= ~led_protocol_error;
                                reset_parser();
                            end else begin
                                header_index <= header_index + 5'd1;
                            end
                        end
                        5'd3: begin
                            if (rx_byte != OPCODE_PING) begin
                                led_protocol_error <= ~led_protocol_error;
                                reset_parser();
                            end else begin
                                header_index <= header_index + 5'd1;
                            end
                        end
                        5'd4: begin
                            sequence[7:0] <= rx_byte;
                            header_index <= header_index + 5'd1;
                        end
                        5'd5: begin
                            sequence[15:8] <= rx_byte;
                            header_index <= header_index + 5'd1;
                        end
                        5'd6: begin
                            sequence[23:16] <= rx_byte;
                            header_index <= header_index + 5'd1;
                        end
                        5'd7: begin
                            sequence[31:24] <= rx_byte;
                            header_index <= header_index + 5'd1;
                        end
                        5'd8: begin
                            payload_len[7:0] <= rx_byte;
                            header_index <= header_index + 5'd1;
                        end
                        5'd9: begin
                            payload_len[15:8] <= rx_byte;
                            header_index <= header_index + 5'd1;
                        end
                        5'd10: begin
                            if (rx_byte != 8'h00) begin
                                led_protocol_error <= ~led_protocol_error;
                                reset_parser();
                            end else begin
                                header_index <= header_index + 5'd1;
                            end
                        end
                        5'd11: begin
                            if (rx_byte != 8'h00) begin
                                led_protocol_error <= ~led_protocol_error;
                                reset_parser();
                            end else begin
                                header_index <= header_index + 5'd1;
                            end
                        end
                        5'd12, 5'd13: begin
                            header_index <= header_index + 5'd1;
                        end
                        5'd14: begin
                            if (rx_byte != 8'h00) begin
                                led_protocol_error <= ~led_protocol_error;
                                reset_parser();
                            end else begin
                                header_index <= header_index + 5'd1;
                            end
                        end
                        5'd15: begin
                            if (rx_byte != 8'h00 || payload_len > MAX_PAYLOAD_LEN) begin
                                led_protocol_error <= ~led_protocol_error;
                                reset_parser();
                            end else begin
                                checksum_index <= 2'd0;
                                if (payload_len == 16'd0) begin
                                    rx_state <= RX_CSUM;
                                end else begin
                                    payload_count <= 16'd0;
                                    rx_state <= RX_PAYLOAD;
                                end
                            end
                        end
                        default: reset_parser();
                    endcase
                end

                RX_PAYLOAD: begin
                    checksum_calc <= fnv_feed(checksum_calc, rx_byte);
                    if (payload_count + 16'd1 == payload_len) begin
                        checksum_index <= 2'd0;
                        rx_state <= RX_CSUM;
                    end
                    payload_count <= payload_count + 16'd1;
                end

                RX_CSUM: begin
                    case (checksum_index)
                        2'd0: begin
                            checksum_seen[7:0] <= rx_byte;
                            checksum_index <= 2'd1;
                        end
                        2'd1: begin
                            checksum_seen[15:8] <= rx_byte;
                            checksum_index <= 2'd2;
                        end
                        2'd2: begin
                            checksum_seen[23:16] <= rx_byte;
                            checksum_index <= 2'd3;
                        end
                        2'd3: begin
                            if (checksum_calc == {rx_byte, checksum_seen[23:0]}) begin
                                pad_remaining <= frame_padding_len(payload_len);
                                if (frame_padding_len(payload_len) == 3'd0) begin
                                    accept_ping();
                                end else begin
                                    rx_state <= RX_PAD;
                                end
                            end else begin
                                led_checksum_error <= ~led_checksum_error;
                                reset_parser();
                            end
                        end
                    endcase
                end

                RX_PAD: begin
                    if (rx_byte != 8'h00) begin
                        led_protocol_error <= ~led_protocol_error;
                        reset_parser();
                    end else if (pad_remaining <= 3'd1) begin
                        accept_ping();
                    end else begin
                        pad_remaining <= pad_remaining - 3'd1;
                    end
                end

                default: reset_parser();
            endcase
        end
    end
endmodule

module uart_rx_8n1 #(
    parameter integer CLKS_PER_BIT = 234,
    parameter integer HALF_CLKS_PER_BIT = 117
) (
    input  wire       clk,
    input  wire       rx,
    output reg  [7:0] data = 8'd0,
    output reg        valid = 1'b0
);
    localparam [2:0] IDLE  = 3'd0;
    localparam [2:0] START = 3'd1;
    localparam [2:0] DATA  = 3'd2;
    localparam [2:0] STOP  = 3'd3;

    reg [2:0]  state = IDLE;
    reg [15:0] clk_count = 16'd0;
    reg [2:0]  bit_index = 3'd0;
    reg [7:0]  rx_shift = 8'd0;
    reg        rx_meta = 1'b1;
    reg        rx_sync = 1'b1;

    always @(posedge clk) begin
        rx_meta <= rx;
        rx_sync <= rx_meta;
        valid <= 1'b0;

        case (state)
            IDLE: begin
                clk_count <= 16'd0;
                bit_index <= 3'd0;
                if (rx_sync == 1'b0) begin
                    state <= START;
                end
            end

            START: begin
                if (clk_count == HALF_CLKS_PER_BIT - 1) begin
                    if (rx_sync == 1'b0) begin
                        clk_count <= 16'd0;
                        state <= DATA;
                    end else begin
                        state <= IDLE;
                    end
                end else begin
                    clk_count <= clk_count + 16'd1;
                end
            end

            DATA: begin
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count <= 16'd0;
                    rx_shift[bit_index] <= rx_sync;
                    if (bit_index == 3'd7) begin
                        bit_index <= 3'd0;
                        state <= STOP;
                    end else begin
                        bit_index <= bit_index + 3'd1;
                    end
                end else begin
                    clk_count <= clk_count + 16'd1;
                end
            end

            STOP: begin
                if (clk_count == CLKS_PER_BIT - 1) begin
                    data <= rx_shift;
                    valid <= rx_sync;
                    clk_count <= 16'd0;
                    state <= IDLE;
                end else begin
                    clk_count <= clk_count + 16'd1;
                end
            end

            default: state <= IDLE;
        endcase
    end
endmodule

module uart_tx_8n1 #(
    parameter integer CLKS_PER_BIT = 234
) (
    input  wire       clk,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx = 1'b1,
    output reg        busy = 1'b0
);
    localparam [2:0] IDLE  = 3'd0;
    localparam [2:0] START = 3'd1;
    localparam [2:0] DATA  = 3'd2;
    localparam [2:0] STOP  = 3'd3;

    reg [2:0]  state = IDLE;
    reg [15:0] clk_count = 16'd0;
    reg [2:0]  bit_index = 3'd0;
    reg [7:0]  tx_shift = 8'd0;

    always @(posedge clk) begin
        case (state)
            IDLE: begin
                tx <= 1'b1;
                busy <= 1'b0;
                clk_count <= 16'd0;
                bit_index <= 3'd0;
                if (start) begin
                    tx_shift <= data;
                    busy <= 1'b1;
                    state <= START;
                end
            end

            START: begin
                tx <= 1'b0;
                busy <= 1'b1;
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count <= 16'd0;
                    state <= DATA;
                end else begin
                    clk_count <= clk_count + 16'd1;
                end
            end

            DATA: begin
                tx <= tx_shift[bit_index];
                busy <= 1'b1;
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count <= 16'd0;
                    if (bit_index == 3'd7) begin
                        bit_index <= 3'd0;
                        state <= STOP;
                    end else begin
                        bit_index <= bit_index + 3'd1;
                    end
                end else begin
                    clk_count <= clk_count + 16'd1;
                end
            end

            STOP: begin
                tx <= 1'b1;
                busy <= 1'b1;
                if (clk_count == CLKS_PER_BIT - 1) begin
                    clk_count <= 16'd0;
                    state <= IDLE;
                end else begin
                    clk_count <= clk_count + 16'd1;
                end
            end

            default: begin
                tx <= 1'b1;
                busy <= 1'b0;
                state <= IDLE;
            end
        endcase
    end
endmodule
